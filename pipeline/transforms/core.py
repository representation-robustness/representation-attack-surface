"""Semantics-preserving source transformations, v3.

Three families, same templates as the paper (identifier renaming, dead-code
insertion, control-flow restructuring), re-implemented to close the validity
holes found in the v1 artifact transforms (tag v1-icse27-submission, utils/obf_transforms_v2.py):

* Opaque predicates only use *parameters* whose declared type is a scalar
  integer (exact token match, plain declarator, not volatile). Parameters are
  always in scope and initialised. Otherwise a literal is used.
* Every inserted or rewritten statement is emitted as its own `{ ... }` block,
  so braceless bodies, dangling-else and case labels cannot change meaning.
* Renaming rewrites *every* occurrence of a local/parameter (including calls
  through local function pointers) and skips names that also appear inside a
  function-local #define or as a type name. Fresh names never collide.
* Calls to ALL_CAPS (macro-like) names are never wrapped or flattened: their
  expansion may declare variables or jump, which cannot be checked statically.
* Edit sites never overlap, so reported edit counts are exact.

Each transform returns (new_source, info) where info records what was done.
A transform that cannot find a function or any site returns the source
unchanged with info["changed"] = False; the validity gate decides what counts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

_PARSERS = {
    "c": Parser(Language(tree_sitter_c.language())),
    "cpp": Parser(Language(tree_sitter_cpp.language())),
}

INT_TYPE_TOKENS = {
    "int", "unsigned", "signed", "short", "long", "char",
    "size_t", "ssize_t", "ptrdiff_t", "off_t", "intptr_t", "uintptr_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64", "__u8", "__u16", "__u32", "__u64",
    "__s8", "__s16", "__s32", "__s64", "uint", "ulong", "ushort", "uchar", "gint", "guint",
}
TYPE_QUALIFIERS_OK = {"const", "register"}
MACRO_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{2,}$")
JUMPS = ("goto_statement", "return_statement", "break_statement", "continue_statement")
PREPROC_COND = ("preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else", "preproc_elifdef")
STATEMENT_TYPES = {
    "expression_statement", "declaration", "if_statement", "while_statement", "for_statement",
    "do_statement", "return_statement", "switch_statement", "compound_statement",
}


# --------------------------------------------------------------------------- parsing

@dataclass
class Parsed:
    src: bytes
    lang: str
    root: Node
    func: Node | None
    body: Node | None
    errors: int


def count_errors(root: Node) -> int:
    n, stack = 0, [root]
    while stack:
        x = stack.pop()
        if x.type == "ERROR" or x.is_missing:
            n += 1
        stack.extend(x.children)
    return n


def _find_function(root: Node) -> Node | None:
    """Largest function_definition with a compound body anywhere in the tree."""
    best, stack = None, [root]
    while stack:
        x = stack.pop()
        if x.type == "function_definition":
            body = x.child_by_field_name("body")
            if body is not None and body.type == "compound_statement":
                if best is None or (x.end_byte - x.start_byte) > (best.end_byte - best.start_byte):
                    best = x
            continue
        stack.extend(x.children)
    return best


def parse(src: str, lang: str | None = None) -> Parsed:
    """Parse with C, fall back to C++ when C finds no function or more errors."""
    b = src.encode("utf-8", errors="replace")
    cands = []
    for lg in ([lang] if lang else ["c", "cpp"]):
        root = _PARSERS[lg].parse(b).root_node
        func = _find_function(root)
        cands.append(Parsed(b, lg, root, func, func.child_by_field_name("body") if func else None,
                            count_errors(root)))
        if lang is None and func is not None and cands[-1].errors == 0:
            break
    # prefer: function found, then fewer errors, then C
    cands.sort(key=lambda p: (p.func is None, p.errors))
    return cands[0]


def text(n: Node, src: bytes) -> str:
    return src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def walk(n: Node):
    stack = [n]
    while stack:
        x = stack.pop()
        yield x
        stack.extend(reversed(x.children))


def has_desc(n: Node, types) -> bool:
    return any(x.type in types for x in walk(n) if x is not n)


def has_anc(n: Node, types, stop: Node | None = None) -> bool:
    p = n.parent
    while p is not None and p is not stop:
        if p.type in types:
            return True
        p = p.parent
    return False


def indent_of(n: Node, src: bytes) -> str:
    line_start = src.rfind(b"\n", 0, n.start_byte) + 1
    lead = src[line_start:n.start_byte]
    return lead.decode("utf-8", "replace") if lead.strip() == b"" else "    "


# --------------------------------------------------------------------------- edits

@dataclass
class Edits:
    reps: list = field(default_factory=list)  # (start, end, bytes)

    def overlaps(self, s: int, e: int) -> bool:
        for a, b, _ in self.reps:
            if s == e == a == b:
                return True
            if s < b and a < e:
                return True
            if s == e and a < s < b:
                return True
        return False

    def add(self, s: int, e: int, new: str) -> bool:
        if self.overlaps(s, e):
            return False
        self.reps.append((s, e, new.encode("utf-8")))
        return True

    def apply(self, src: bytes) -> bytes:
        out = bytearray(src)
        for s, e, new in sorted(self.reps, key=lambda r: (r[0], r[1]), reverse=True):
            out[s:e] = new
        return bytes(out)


class Fresh:
    """Collision-free fresh identifiers of the form __<prefix>_NNNN."""

    def __init__(self, src: bytes):
        self.taken = set(re.findall(rb"[A-Za-z_][A-Za-z0-9_]*", src))
        self.ctr = 0

    def __call__(self, prefix: str) -> str:
        while True:
            self.ctr += 1
            name = f"__{prefix}_{self.ctr:04d}"
            if name.encode() not in self.taken:
                self.taken.add(name.encode())
                return name


# --------------------------------------------------------------------------- analysis helpers

def declarator_name(d: Node | None, src: bytes) -> tuple[str | None, bool]:
    """(name, is_plain) for a declarator; plain = bare identifier (no *, [], ())."""
    plain = True
    while d is not None:
        if d.type in ("identifier", "field_identifier"):
            return text(d, src), plain
        if d.type in ("pointer_declarator", "array_declarator", "function_declarator",
                      "reference_declarator", "abstract_pointer_declarator"):
            plain = False
        if d.type == "parenthesized_declarator":
            d = d.named_children[0] if d.named_children else None
            continue
        nxt = d.child_by_field_name("declarator")
        if nxt is None and d.type == "reference_declarator" and d.named_children:
            nxt = d.named_children[-1]
        d = nxt
    return None, plain


def param_decls(func: Node) -> list[Node]:
    out = []
    decl = func.child_by_field_name("declarator")
    while decl is not None and decl.type != "function_declarator":
        decl = decl.child_by_field_name("declarator")
    if decl is None:
        return out
    plist = decl.child_by_field_name("parameters")
    if plist is None:
        return out
    return [p for p in plist.named_children if p.type in ("parameter_declaration", "optional_parameter_declaration")]


def int_params(p: Parsed) -> list[str]:
    """Parameters that are scalar integers by exact type tokens, plain declarator, not volatile."""
    out = []
    for prm in param_decls(p.func):
        decl = prm.child_by_field_name("declarator")
        if prm.child_by_field_name("type") is None or decl is None:
            continue
        # every word before the declarator: qualifiers + type specifiers
        type_part = p.src[prm.start_byte:decl.start_byte].decode("utf-8", "replace")
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", type_part))
        if "volatile" in words or not words:
            continue
        if not (words - TYPE_QUALIFIERS_OK) <= INT_TYPE_TOKENS:
            continue
        name, plain = declarator_name(decl, p.src)
        if name and plain:
            out.append(name)
    return out


def is_macro_call(call: Node, src: bytes) -> bool:
    fn = call.child_by_field_name("function")
    return fn is not None and fn.type == "identifier" and bool(MACRO_NAME.match(text(fn, src)))


def in_preproc(n: Node, stop: Node) -> bool:
    """Inside a preprocessor conditional, or inside a region tree-sitter could not parse."""
    return has_anc(n, PREPROC_COND + ("ERROR",), stop) or n.type == "ERROR"


def is_macro_header(n: Node | None) -> bool:
    """Expression statement with no real ';' — e.g. `list_for_each(p, h)` heading the
    block or statement that follows it. Rewriting it or inserting after it would
    detach the macro from its body."""
    if n is None or n.type != "expression_statement":
        return False
    last = n.children[-1] if n.children else None
    return last is None or last.type != ";" or last.is_missing


# --------------------------------------------------------------------------- 1. identifier renaming

def rename_identifiers(src: str, lang: str | None = None) -> tuple[str, dict]:
    p = parse(src, lang)
    info = {"transform": "identifier", "lang": p.lang, "func_found": p.func is not None, "changed": False}
    if p.func is None:
        return src, info

    targets: set[str] = set()
    for prm in param_decls(p.func):
        name, _ = declarator_name(prm.child_by_field_name("declarator"), p.src)
        if name:
            targets.add(name)
    for d in walk(p.body):
        if d.type == "declaration":
            for c in d.children_by_field_name("declarator"):
                name, _ = declarator_name(c, p.src)
                if name:
                    targets.add(name)

    # unsafe names: used in a function-local #define body or also used as a type
    macro_text = " ".join(text(x, p.src) for x in walk(p.func)
                          if x.type in ("preproc_def", "preproc_function_def", "preproc_call"))
    type_names = {text(x, p.src) for x in walk(p.func) if x.type in ("type_identifier", "primitive_type")}
    skipped = {t for t in targets if re.search(rf"\b{re.escape(t)}\b", macro_text) or t in type_names}
    targets -= skipped
    if not targets:
        info["skipped_names"] = sorted(skipped)
        return src, info

    fresh = Fresh(p.src)
    mapping = {t: fresh("v") for t in sorted(targets)}
    edits = Edits()
    n_occ = 0
    for x in walk(p.func):
        if x.type == "identifier" and text(x, p.src) in mapping:
            par = x.parent
            if par is not None and par.type in ("goto_statement", "labeled_statement"):
                continue
            edits.add(x.start_byte, x.end_byte, mapping[text(x, p.src)])
            n_occ += 1
    new = edits.apply(p.src).decode("utf-8", errors="replace")
    info.update(changed=new != src, n_names=len(mapping), n_occurrences=n_occ,
                skipped_names=sorted(skipped), mapping=mapping)
    return new, info


# --------------------------------------------------------------------------- 2. dead-code insertion

def _insertion_points(p: Parsed) -> list[Node]:
    """Statements that are direct children of a compound statement, outside #if blocks."""
    pts = []
    for x in walk(p.body):
        if x.type == "compound_statement" and not in_preproc(x, p.func):
            prev = None
            for c in x.named_children:
                if c.type in STATEMENT_TYPES or c.type.endswith("_statement"):
                    if c.type not in ("labeled_statement", "case_statement", "comment") \
                            and not is_macro_header(prev) and not is_macro_header(c):
                        pts.append(c)
                if c.type != "comment":
                    prev = c
    pts.sort(key=lambda n: n.start_byte)
    return pts


def insert_dead_code(src: str, lang: str | None = None, max_sites: int = 7) -> tuple[str, dict]:
    p = parse(src, lang)
    info = {"transform": "deadcode", "lang": p.lang, "func_found": p.func is not None, "changed": False}
    if p.func is None:
        return src, info
    pts = _insertion_points(p)
    if len(pts) < 2:
        info["reason"] = "fewer than 2 insertion points"
        return src, info
    ivars = int_params(p)
    var = ivars[0] if ivars else None
    step = max(1, len(pts) // max_sites)
    chosen = pts[::step][:max_sites]

    fresh = Fresh(p.src)
    edits = Edits()
    added_tokens = 0
    for i, st in enumerate(chosen):
        ind = indent_of(st, p.src)
        if i % 3 == 2:
            dm, fl = fresh("dm"), fresh("fl")
            block = (f"{{ int {dm} = 0; for (int {fl} = 0; {fl} < 0; {fl}++) {{ {dm} += {fl}; }} }}\n{ind}")
        else:
            dc = fresh("dc")
            cond = f"({var} ^ {var})" if var else "0"
            block = f"{{ int {dc} = {cond}; if ({dc} != 0) {{ {dc} = {dc} + 1; }} }}\n{ind}"
        if edits.add(st.start_byte, st.start_byte, block):
            added_tokens += len(re.findall(r"\w+|[^\w\s]", block))
    new = edits.apply(p.src).decode("utf-8", errors="replace")
    info.update(changed=new != src, n_sites=len(edits.reps), opaque_var=var,
                tokens_added=added_tokens)
    return new, info


# --------------------------------------------------------------------------- 3. control-flow restructuring

def _cond_text(cond: Node, src: bytes) -> str | None:
    """Text of an if/while condition without its outer parens; None if it declares something."""
    if has_desc(cond, ("declaration", "init_declarator", "condition_clause")) and cond.type != "parenthesized_expression":
        return None
    if cond.type == "condition_clause":
        inner = [c for c in cond.named_children]
        if len(inner) != 1 or inner[0].type in ("declaration", "init_statement"):
            return None
        return text(inner[0], src)
    t = text(cond, src).strip()
    return t[1:-1] if t.startswith("(") and t.endswith(")") else t


def _braced(n: Node, src: bytes) -> str:
    t = text(n, src).strip()
    return t if n.type == "compound_statement" else f"{{ {t} }}"


def restructure_control_flow(src: str, lang: str | None = None) -> tuple[str, dict]:
    p = parse(src, lang)
    info = {"transform": "controlflow", "lang": p.lang, "func_found": p.func is not None, "changed": False}
    if p.func is None:
        return src, info
    ivars = int_params(p)
    v = ivars[0] if ivars else None
    fresh = Fresh(p.src)
    edits = Edits()
    counts = {"branch_split": 0, "while_opaque": 0, "call_wrap": 0, "flatten": 0, "fallback": 0}

    # 1. branch splitting (<=3, outermost first, no overlap)
    for x in walk(p.body):
        if counts["branch_split"] >= 3:
            break
        if x.type != "if_statement" or in_preproc(x, p.func):
            continue
        cond, cons, alt = (x.child_by_field_name(k) for k in ("condition", "consequence", "alternative"))
        if cond is None or cons is None or alt is None:
            continue
        if alt.type == "else_clause":
            alt = alt.named_children[-1] if alt.named_children else None
        if alt is None or alt.type == "if_statement":
            continue
        if has_desc(x, ("labeled_statement", "case_statement", "preproc_if", "preproc_ifdef")):
            continue
        ct = _cond_text(cond, p.src)
        if ct is None:
            continue
        pv = fresh("p")
        ind = indent_of(x, p.src)
        new = (f"{{ int {pv} = ({ct}) ? 1 : 0;\n{ind}if ({pv}) {_braced(cons, p.src)}\n"
               f"{ind}if (!{pv}) {_braced(alt, p.src)} }}")
        if edits.add(x.start_byte, x.end_byte, new):
            counts["branch_split"] += 1

    # 2. opaque conjunct in while conditions (<=2)
    opaque_true = f"(({v} ^ {v}) == 0)" if v else "(1 == 1)"
    for x in walk(p.body):
        if counts["while_opaque"] >= 2:
            break
        if x.type != "while_statement" or in_preproc(x, p.func):
            continue
        cond = x.child_by_field_name("condition")
        ct = _cond_text(cond, p.src) if cond is not None else None
        if ct is None:
            continue
        if edits.add(cond.start_byte, cond.end_byte, f"(({ct}) && {opaque_true})"):
            counts["while_opaque"] += 1

    # 3. wrap every 3rd plain call statement in an always-true predicate (<=3)
    always = f"(({v} | {v}) == {v})" if v else "(1)"
    calls = [x for x in walk(p.body) if x.type == "expression_statement" and not in_preproc(x, p.func)]
    for i, x in enumerate(calls):
        if counts["call_wrap"] >= 3:
            break
        if i % 3 != 0 or not x.named_children or x.named_children[0].type != "call_expression":
            continue
        if is_macro_call(x.named_children[0], p.src) or is_macro_header(x):
            continue
        if edits.add(x.start_byte, x.end_byte, f"{{ if ({always}) {{ {text(x, p.src).strip()} }} }}"):
            counts["call_wrap"] += 1

    # 4. flatten one run of 3-4 consecutive safe expression statements into a dispatcher
    unsafe = set(JUMPS) | {"labeled_statement", "case_statement"} | set(PREPROC_COND)
    for comp in [x for x in walk(p.body) if x.type == "compound_statement"]:
        if counts["flatten"] or in_preproc(comp, p.func):
            continue
        kids = [c for c in comp.named_children if c.type != "comment"]
        for s in range(len(kids) - 2):
            run = []
            if s > 0 and is_macro_header(kids[s - 1]):
                continue
            for c in kids[s:s + 4]:
                if c.type != "expression_statement" or has_desc(c, unsafe) or is_macro_header(c):
                    break
                if any(y.type == "call_expression" and is_macro_call(y, p.src) for y in walk(c)):
                    break
                run.append(c)
            if len(run) < 3:
                continue
            st = fresh("st")
            ind = indent_of(run[0], p.src)
            arms = [f"if ({st} == {k}) {{ {text(c, p.src).strip()} {st} = {k + 1}; }}" for k, c in enumerate(run)]
            new = (f"{{ int {st} = 0;\n{ind}while ({st} < {len(run)}) {{\n{ind}    "
                   + f"\n{ind}    else ".join(arms) + f"\n{ind}}} }}")
            if edits.add(run[0].start_byte, run[-1].end_byte, new):
                counts["flatten"] += 1
                break

    # 5. fallback: one structural opaque predicate before the first statement
    if not edits.reps:
        pts = _insertion_points(p)
        if pts:
            pv = fresh("p")
            edits.add(pts[0].start_byte, pts[0].start_byte,
                      f"{{ int {pv} = (0 == 0); if ({pv} && ({pv} == 1)) {{ {pv} = {pv}; }} }}\n{indent_of(pts[0], p.src)}")
            counts["fallback"] = 1

    new = edits.apply(p.src).decode("utf-8", errors="replace")
    info.update(changed=new != src, opaque_var=v, n_sites=len(edits.reps), **counts)
    return new, info


# --------------------------------------------------------------------------- composition

TRANSFORMS: dict[str, Callable] = {
    "identifier": rename_identifiers,
    "deadcode": insert_dead_code,
    "controlflow": restructure_control_flow,
}

CONDITIONS: dict[str, list[str]] = {
    "identifier": ["identifier"],
    "deadcode": ["deadcode"],
    "controlflow": ["controlflow"],
    "ren_dead": ["identifier", "deadcode"],
    "ren_cf": ["identifier", "controlflow"],
    "dead_cf": ["deadcode", "controlflow"],
    "compound": ["identifier", "deadcode", "controlflow"],
}


def apply_condition(src: str, condition: str) -> tuple[str, list[dict]]:
    """Apply a condition's transforms in order; the language is fixed by the first parse."""
    lang = parse(src).lang
    infos, cur = [], src
    for name in CONDITIONS[condition]:
        try:
            cur, info = TRANSFORMS[name](cur, lang)
        except Exception as e:  # recorded, never silently swallowed
            info = {"transform": name, "error": f"{type(e).__name__}: {e}", "changed": False}
        infos.append(info)
    return cur, infos
