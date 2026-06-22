#!/usr/bin/env python3
"""
Strong semantics-preserving obfuscation transformer for devign_full.

Three independent modes:
  identifier  – rename local vars and parameters to __v_NNNN
  deadcode    – insert opaque predicates and fake loops throughout
  controlflow – branch splitting, loop opaque predicates, call wrapping, block flattening

Usage (run from thesis root or devign_full/):
  python devign_full/obf_transforms_v2.py --obf identifier
  python devign_full/obf_transforms_v2.py --obf deadcode
  python devign_full/obf_transforms_v2.py --obf controlflow
  python devign_full/obf_transforms_v2.py --obf identifier --test 10
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Set, Tuple

import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

C_LANGUAGE = Language(tsc.language())
_parser = Parser(C_LANGUAGE)

THESIS_ROOT = Path(__file__).resolve().parent.parent
ORIGINALS_DIR = Path(__file__).resolve().parent / "originals"


# ---------------------------------------------------------------------------
# Unique name counter (reset per file)
# ---------------------------------------------------------------------------

_ctr = 0


def _fresh(prefix: str) -> str:
    global _ctr
    _ctr += 1
    return f"__{prefix}_{_ctr:04d}"


def _reset_ctr():
    global _ctr
    _ctr = 0


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------

def parse_src(src: str) -> Node:
    return _parser.parse(src.encode("utf-8", errors="replace")).root_node


def node_text(node: Node, src_bytes: bytes) -> str:
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def find_nodes(root: Node, *types: str) -> List[Node]:
    """DFS collect all nodes of given types anywhere in the subtree."""
    result = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in types:
            result.append(n)
        stack.extend(reversed(n.children))
    return result


def has_descendant(node: Node, *types: str) -> bool:
    """True if any descendant has one of the given types."""
    stack = list(node.children)
    while stack:
        n = stack.pop()
        if n.type in types:
            return True
        stack.extend(n.children)
    return False


def has_ancestor(node: Node, *types: str) -> bool:
    """True if any ancestor has one of the given types."""
    parent = node.parent
    while parent:
        if parent.type in types:
            return True
        parent = parent.parent
    return False


def get_function_def(root: Node) -> Optional[Node]:
    for node in root.named_children:
        if node.type == "function_definition":
            return node
    return None


def get_body(func_def: Node) -> Optional[Node]:
    body = func_def.child_by_field_name("body")
    return body if body and body.type == "compound_statement" else None


# ---------------------------------------------------------------------------
# Replacement engine
# ---------------------------------------------------------------------------

Replacement = Tuple[int, int, bytes]


def apply_replacements(src_bytes: bytes, reps: List[Replacement]) -> bytes:
    """
    Apply (start, end, new_bytes) replacements.
    Overlapping replacements are resolved by keeping the first one
    (stable sort by start ensures determinism).
    Applied in reverse order to preserve byte offsets.
    """
    if not reps:
        return src_bytes

    # Sort by start byte; stable so first-added wins on tie
    reps.sort(key=lambda x: x[0])

    # Remove overlaps: keep a rep only if it starts at or after previous end
    merged: List[Replacement] = []
    last_end = -1
    for start, end, text in reps:
        if start >= last_end:
            merged.append((start, end, text))
            last_end = max(last_end, end)

    # Apply in reverse to preserve offsets
    merged.sort(key=lambda x: x[0], reverse=True)
    buf = bytearray(src_bytes)
    for start, end, text in merged:
        buf[start:end] = text
    return bytes(buf)


# ---------------------------------------------------------------------------
# Identifier collection utilities
# ---------------------------------------------------------------------------

def get_declared_name(declarator: Node, src_bytes: bytes) -> Optional[str]:
    """Recursively unwrap a declarator node to get the variable name."""
    if declarator is None:
        return None
    if declarator.type == "identifier":
        return node_text(declarator, src_bytes)
    if declarator.type in (
        "pointer_declarator", "array_declarator",
        "function_declarator", "init_declarator",
        "parenthesized_declarator",
    ):
        inner = declarator.child_by_field_name("declarator")
        return get_declared_name(inner, src_bytes)
    return None


def collect_param_names(func_def: Node, src_bytes: bytes) -> Set[str]:
    """Parameter names from the function signature."""
    names: Set[str] = set()
    declarator = func_def.child_by_field_name("declarator")
    if not declarator:
        return names
    for pl in find_nodes(declarator, "parameter_list"):
        for param in pl.named_children:
            if param.type == "parameter_declaration":
                d = param.child_by_field_name("declarator")
                name = get_declared_name(d, src_bytes)
                if name:
                    names.add(name)
    return names


def collect_local_names(body: Node, src_bytes: bytes) -> Set[str]:
    """Names of all locally declared variables inside the function body."""
    names: Set[str] = set()
    for decl in find_nodes(body, "declaration"):
        # Skip declarations inside preprocessor conditionals for safety
        if has_ancestor(decl, "preproc_if", "preproc_ifdef", "preproc_elif"):
            continue
        # A declaration can have multiple declarators (int a, b, c;)
        for child in decl.named_children:
            if child.type in (
                "identifier", "init_declarator", "pointer_declarator",
                "array_declarator", "function_declarator",
            ):
                # Skip the type child (first named child is often the type)
                if child == decl.child_by_field_name("type"):
                    continue
                name = get_declared_name(child, src_bytes)
                if name:
                    names.add(name)
    return names


def is_field_ref(node: Node) -> bool:
    """True if this identifier is the field part of a struct access (->field or .field)."""
    return node.type == "field_identifier"


def is_call_function(node: Node) -> bool:
    """True if this identifier is the function being called, not an argument."""
    p = node.parent
    if p and p.type == "call_expression":
        fn = p.child_by_field_name("function")
        return fn is not None and fn.start_byte == node.start_byte and fn.end_byte == node.end_byte
    return False


def is_goto_or_label(node: Node) -> bool:
    p = node.parent
    if p is None:
        return False
    return p.type in ("goto_statement", "labeled_statement")


# ---------------------------------------------------------------------------
# Int-var finder (for opaque predicates that use existing variables)
# ---------------------------------------------------------------------------

INT_LIKE = ("int", "uint", "long", "size_t", "char", "short", "cl_int", "cl_uint")


def find_int_params(func_def: Node, src_bytes: bytes) -> List[str]:
    """Return names of int-like parameters (always in scope, always initialized)."""
    result = []
    declarator = func_def.child_by_field_name("declarator")
    if not declarator:
        return result
    for pl in find_nodes(declarator, "parameter_list"):
        for param in pl.named_children:
            if param.type != "parameter_declaration":
                continue
            type_node = param.child_by_field_name("type")
            if not type_node:
                continue
            type_text = node_text(type_node, src_bytes)
            if not any(t in type_text for t in INT_LIKE):
                continue
            d = param.child_by_field_name("declarator")
            name = get_declared_name(d, src_bytes)
            if name:
                result.append(name)
    return result


def find_int_vars(body: Node, src_bytes: bytes) -> List[str]:
    """Return names of locally declared int-like variables (safe to use in XOR predicates)."""
    result = []
    for decl in find_nodes(body, "declaration"):
        if has_ancestor(decl, "preproc_if", "preproc_ifdef"):
            continue
        type_node = decl.child_by_field_name("type")
        if not type_node:
            continue
        type_text = node_text(type_node, src_bytes)
        if not any(t in type_text for t in INT_LIKE):
            continue
        for child in decl.named_children:
            if child == type_node:
                continue
            name = get_declared_name(child, src_bytes)
            if name:
                result.append(name)
    return result


# ===========================================================================
# OBFUSCATION TYPE 1: IDENTIFIER RENAMING
# ===========================================================================

def obf_identifier(src: str) -> str:
    _reset_ctr()
    src_bytes = src.encode("utf-8", errors="replace")
    root = parse_src(src)

    func_def = get_function_def(root)
    if not func_def:
        return src

    body = get_body(func_def)
    if not body:
        return src

    # Build rename map: only params + locals
    targets = collect_param_names(func_def, src_bytes) | collect_local_names(body, src_bytes)
    if not targets:
        return src

    rename = {name: _fresh("v") for name in sorted(targets)}

    # Find every `identifier` node in the function and rename if in map
    reps: List[Replacement] = []
    for node in find_nodes(func_def, "identifier"):
        name = node_text(node, src_bytes)
        if name not in rename:
            continue
        if is_call_function(node):
            continue
        if is_goto_or_label(node):
            continue
        reps.append((node.start_byte, node.end_byte, rename[name].encode()))

    return apply_replacements(src_bytes, reps).decode("utf-8", errors="replace")


# ===========================================================================
# OBFUSCATION TYPE 1b: VOCABULARY-SHIFT RENAMING
# Rename to rare/OOV identifiers with unusual subword patterns.
# The graph topology is identical to obf_identifier; only node tokens change.
# ===========================================================================

# Rare consonant clusters that are valid C identifiers but extremely uncommon
_VOCAB_SHIFT_PREFIXES = [
    "xqz", "vwb", "zxk", "qvj", "bjx", "fqz", "xwv", "zgk", "qbz", "vxj",
    "jzq", "wqv", "kzx", "xzg", "bzv", "qjw", "vzx", "xkq", "zqb", "jwx",
]


def obf_vocab_shift(src: str) -> str:
    """Rename locals/params to OOV-style identifiers (rare consonant clusters)."""
    _reset_ctr()
    src_bytes = src.encode("utf-8", errors="replace")
    root = parse_src(src)

    func_def = get_function_def(root)
    if not func_def:
        return src
    body = get_body(func_def)
    if not body:
        return src

    targets = collect_param_names(func_def, src_bytes) | collect_local_names(body, src_bytes)
    if not targets:
        return src

    prefix_idx = 0
    rename: dict = {}
    for name in sorted(targets):
        global _ctr
        _ctr += 1
        pfx = _VOCAB_SHIFT_PREFIXES[prefix_idx % len(_VOCAB_SHIFT_PREFIXES)]
        prefix_idx += 1
        rename[name] = f"__{pfx}_{_ctr:04d}"

    reps: List[Replacement] = []
    for node in find_nodes(func_def, "identifier"):
        name = node_text(node, src_bytes)
        if name not in rename:
            continue
        if is_call_function(node):
            continue
        if is_goto_or_label(node):
            continue
        reps.append((node.start_byte, node.end_byte, rename[name].encode()))

    return apply_replacements(src_bytes, reps).decode("utf-8", errors="replace")


# ===========================================================================
# OBFUSCATION TYPE 1c: BENIGN-TOKEN RENAMING
# Rename locals/params to safety-suggesting names.
# Hypothesis: transformer models may assign lower risk to functions whose
# identifiers semantically suggest safety, bounds-checking, or validation.
# ===========================================================================

_BENIGN_NAMES = [
    "checked_len", "safe_ptr", "validated_idx", "secure_buf",
    "clean_data", "verified_size", "sanitized_val", "trusted_input",
    "auth_guard", "guarded_ptr", "safe_offset", "clean_flag",
    "checked_buf", "valid_size", "safe_count", "verified_ptr",
    "bounded_idx", "safe_len", "checked_val", "guard_flag",
    "safe_index", "clean_ptr", "validated_len", "secure_size",
]


def obf_benign_token(src: str) -> str:
    """Rename locals/params to safety-suggesting names."""
    _reset_ctr()
    src_bytes = src.encode("utf-8", errors="replace")
    root = parse_src(src)

    func_def = get_function_def(root)
    if not func_def:
        return src
    body = get_body(func_def)
    if not body:
        return src

    targets = collect_param_names(func_def, src_bytes) | collect_local_names(body, src_bytes)
    if not targets:
        return src

    rename: dict = {}
    for i, name in enumerate(sorted(targets)):
        rename[name] = _BENIGN_NAMES[i % len(_BENIGN_NAMES)]

    reps: List[Replacement] = []
    for node in find_nodes(func_def, "identifier"):
        name = node_text(node, src_bytes)
        if name not in rename:
            continue
        if is_call_function(node):
            continue
        if is_goto_or_label(node):
            continue
        reps.append((node.start_byte, node.end_byte, rename[name].encode()))

    return apply_replacements(src_bytes, reps).decode("utf-8", errors="replace")


# ===========================================================================
# OBFUSCATION TYPE 4: TEMPORARY-VARIABLE INTRODUCTION
# Extract binary/ternary subexpressions used as function-call arguments into
# freshly declared temp variables.  This adds new data-dependency edges to the
# CPG without altering observable behaviour.
# ===========================================================================

_EXTRACTABLE_EXPRS = frozenset({
    "binary_expression", "conditional_expression", "cast_expression",
    "unary_expression",
})

_SAFE_CAST_LEAF = frozenset({
    "number_literal", "identifier", "string_literal",
    "char_literal", "true", "false", "null",
})


def _collect_call_args(body: Node, src_bytes: bytes) -> List[Tuple[Node, List[Node]]]:
    """Return (call_expr_node, [extractable_arg_nodes]) for each call in body."""
    result = []
    for call in find_nodes(body, "call_expression"):
        if has_ancestor(call, "preproc_if", "preproc_ifdef"):
            continue
        args_node = call.child_by_field_name("arguments")
        if not args_node:
            continue
        extractable = []
        for arg in args_node.named_children:
            if arg.type in _EXTRACTABLE_EXPRS:
                # Don't extract single-leaf unary (e.g. `&x`, `*p`) — too risky
                if arg.type == "unary_expression":
                    op = arg.child_by_field_name("operator")
                    if op and node_text(op, src_bytes) in ("&", "*"):
                        continue
                extractable.append(arg)
        if extractable:
            result.append((call, extractable))
    return result


def obf_tempvar(src: str) -> str:
    """Extract complex call arguments to temporary variables."""
    _reset_ctr()
    src_bytes = src.encode("utf-8", errors="replace")
    root = parse_src(src)

    func_def = get_function_def(root)
    if not func_def:
        return src
    body = get_body(func_def)
    if not body:
        return src

    call_args = _collect_call_args(body, src_bytes)
    if not call_args:
        return src

    reps: List[Replacement] = []
    # At most 5 call sites to avoid excessive code bloat
    for call_node, args in call_args[:5]:
        # Find the enclosing statement (direct child of body or a compound block)
        stmt = call_node
        while stmt.parent and stmt.parent.type not in ("compound_statement",):
            stmt = stmt.parent

        if stmt.parent is None or stmt.parent.type != "compound_statement":
            continue

        # Replace each extractable arg with a fresh tmp variable
        decls: List[str] = []
        arg_reps: List[Replacement] = []
        for arg in args[:4]:  # at most 4 args per call
            expr_txt = node_text(arg, src_bytes)
            tmp = _fresh("tmp")
            decls.append(f"int {tmp} = {expr_txt};")
            arg_reps.append((arg.start_byte, arg.end_byte, tmp.encode()))

        if not decls:
            continue

        # Insert declarations before the enclosing statement
        decl_block = "\n    ".join(decls) + "\n    "
        reps.append((stmt.start_byte, stmt.start_byte, decl_block.encode()))
        reps.extend(arg_reps)

    return apply_replacements(src_bytes, reps).decode("utf-8", errors="replace")


# ===========================================================================
# OBFUSCATION TYPE 2: DEAD CODE INSERTION
# ===========================================================================

STMT_TYPES = frozenset({
    "expression_statement", "declaration", "if_statement",
    "while_statement", "for_statement", "return_statement",
    "do_statement", "break_statement", "continue_statement",
})


def collect_stmt_insertion_points(body: Node, src_bytes: bytes) -> List[int]:
    """
    Collect byte positions (start of statements) suitable for dead code insertion.
    Avoids preprocessor blocks. Returns positions distributed through the function.
    """
    positions = []

    def recurse(node: Node):
        if node.type in ("preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else"):
            return
        if node.type == "compound_statement":
            children = [c for c in node.named_children if c.type in STMT_TYPES]
            for child in children:
                positions.append(child.start_byte)
                # Recurse into blocks but not into preproc
                if child.type in ("if_statement", "while_statement", "for_statement", "do_statement"):
                    for sub in child.children:
                        if sub.type == "compound_statement":
                            recurse(sub)

    recurse(body)
    return positions


def make_opaque_pred(var: Optional[str], idx: int, indent: str = "    ") -> str:
    dc = f"__dc_{idx:04d}"
    cond = f"({var} ^ {var})" if var else "0"
    return (
        f"{indent}int {dc} = {cond};\n"
        f"{indent}if ({dc} != 0) {{ {dc} = {dc} + 1; }}\n"
    )


def make_fake_loop(idx: int, indent: str = "    ") -> str:
    fl = f"__fl_{idx:04d}"
    dm = f"__dm_{idx:04d}"
    return (
        f"{indent}int {dm} = 0;\n"
        f"{indent}for (int {fl} = 0; {fl} < 0; {fl}++) {{ {dm} += {fl}; }}\n"
    )


def obf_deadcode(src: str) -> str:
    _reset_ctr()
    src_bytes = src.encode("utf-8", errors="replace")
    root = parse_src(src)

    func_def = get_function_def(root)
    if not func_def:
        return src

    body = get_body(func_def)
    if not body:
        return src

    positions = collect_stmt_insertion_points(body, src_bytes)
    if len(positions) < 2:
        return src

    # Prefer params (always in scope) over locals for the XOR variable
    int_params = find_int_params(func_def, src_bytes)
    int_var = int_params[0] if int_params else None

    # Pick ~6 evenly distributed insertion points
    n = len(positions)
    step = max(1, n // 7)
    chosen = positions[::step][:7]

    reps: List[Replacement] = []
    for i, pos in enumerate(chosen):
        if i % 3 == 2:
            text = make_fake_loop(i)
        else:
            text = make_opaque_pred(int_var, i)
        # Insert BEFORE the statement at this byte position
        reps.append((pos, pos, text.encode()))

    return apply_replacements(src_bytes, reps).decode("utf-8", errors="replace")


# ===========================================================================
# OBFUSCATION TYPE 3: CONTROL FLOW
# ===========================================================================

def split_if_else(node: Node, src_bytes: bytes) -> Optional[str]:
    """
    if (cond) { A } else { B }
      →
    int __p_N = (cond) ? 1 : 0;
    if (__p_N) { A }
    if (!__p_N) { B }
    """
    if node.type != "if_statement":
        return None

    cond = node.child_by_field_name("condition")
    cons = node.child_by_field_name("consequence")
    alt = node.child_by_field_name("alternative")

    if not cond or not cons or not alt:
        return None
    # Skip else-if chains (too complex to flatten safely)
    if alt.type == "if_statement":
        return None
    # tree-sitter wraps the else body in an else_clause node; skip else-if inside it
    if alt.type == "else_clause":
        inner = alt.named_children[-1] if alt.named_children else None
        if inner is None:
            return None
        if inner.type == "if_statement":
            return None  # else-if chain
        alt = inner  # use just the body, dropping the 'else' keyword text
    # Skip if either branch contains goto (would break jump targets)
    if has_descendant(cons, "goto_statement") or has_descendant(alt, "goto_statement"):
        return None

    cond_txt = node_text(cond, src_bytes)
    cons_txt = node_text(cons, src_bytes)
    alt_txt = node_text(alt, src_bytes)

    # Ensure braces
    if not cons_txt.strip().startswith("{"):
        cons_txt = f"{{ {cons_txt.strip()} }}"
    if not alt_txt.strip().startswith("{"):
        alt_txt = f"{{ {alt_txt.strip()} }}"

    p = _fresh("p")
    return (
        f"int {p} = ({cond_txt}) ? 1 : 0;\n"
        f"    if ({p}) {cons_txt}\n"
        f"    if (!{p}) {alt_txt}"
    )


def patch_while_condition(node: Node, src_bytes: bytes, int_vars: List[str]) -> Optional[str]:
    """
    while (cond)  →  while ((cond) && ((var ^ var) == 0))
    """
    if node.type != "while_statement":
        return None

    cond = node.child_by_field_name("condition")
    body = node.child_by_field_name("body")
    if not cond or not body:
        return None

    cond_txt = node_text(cond, src_bytes)
    body_txt = node_text(body, src_bytes)

    if int_vars:
        v = int_vars[0]
        opaque = f"(({v} ^ {v}) == 0)"
    else:
        opaque = "(1 == 1)"

    return f"while (({cond_txt}) && {opaque}) {body_txt}"


def wrap_expr_stmt_with_predicate(node: Node, src_bytes: bytes, int_vars: List[str]) -> Optional[str]:
    """
    call(x);  →  if (((x | x) == x)) { call(x); }
    Only wraps plain expression statements that are function calls.
    """
    if node.type != "expression_statement":
        return None
    # Must be a direct call expression (not assignment with call on RHS)
    named = node.named_children
    if not named or named[0].type != "call_expression":
        return None
    # Skip if inside a preproc block
    if has_ancestor(node, "preproc_if", "preproc_ifdef"):
        return None

    stmt_txt = node_text(node, src_bytes).strip()

    if int_vars:
        v = int_vars[0]
        pred = f"(({v} | {v}) == {v})"
    else:
        pred = "(1)"

    return f"if ({pred}) {{\n        {stmt_txt}\n    }}"


def flatten_sequential_block(
    direct_stmts: List[Node], src_bytes: bytes
) -> Optional[Replacement]:
    """
    Find 3-4 consecutive safe statements and replace with a __state dispatcher:

    stmt_A; stmt_B; stmt_C;
      →
    int __st_N = 0;
    while (__st_N < 3) {
        if (__st_N == 0) { stmt_A; __st_N = 1; }
        else if (__st_N == 1) { stmt_B; __st_N = 2; }
        else { stmt_C; __st_N = 3; }
    }

    Safe = expression_statement or simple declaration,
           no return/goto/break/continue/label anywhere inside.
    """
    # Only expression_statement — NOT declarations, since scoping variables
    # into the while body would break any later use of those variables.
    SAFE = frozenset({"expression_statement"})
    UNSAFE_DESCENDANTS = frozenset({
        "goto_statement", "return_statement",
        "break_statement", "continue_statement",
        "labeled_statement", "preproc_if", "preproc_ifdef",
    })

    for start_i in range(len(direct_stmts) - 2):
        run: List[Node] = []
        for j in range(start_i, min(start_i + 4, len(direct_stmts))):
            s = direct_stmts[j]
            if s.type not in SAFE:
                break
            if has_descendant(s, *UNSAFE_DESCENDANTS):
                break
            if has_ancestor(s, "preproc_if", "preproc_ifdef"):
                break
            run.append(s)
        if len(run) < 3:
            continue

        st = _fresh("st")
        n = len(run)
        branches = []
        for k, stmt in enumerate(run):
            txt = node_text(stmt, src_bytes).strip()
            branches.append(f"if ({st} == {k}) {{ {txt} {st} = {k + 1}; }}")

        flat = (
            f"int {st} = 0;\n"
            f"    while ({st} < {n}) {{\n        "
            + "\n        else ".join(branches)
            + f"\n    }}"
        )

        return (run[0].start_byte, run[-1].end_byte, flat.encode())

    return None


def obf_controlflow(src: str) -> str:
    _reset_ctr()
    src_bytes = src.encode("utf-8", errors="replace")
    root = parse_src(src)

    func_def = get_function_def(root)
    if not func_def:
        return src

    body = get_body(func_def)
    if not body:
        return src

    # Prefer params (always in scope) then fall back to locals
    int_vars = find_int_params(func_def, src_bytes) or find_int_vars(body, src_bytes)
    reps: List[Replacement] = []

    # 1. Branch splitting: up to 3 if/else statements
    split_done = 0
    for node in find_nodes(body, "if_statement"):
        if split_done >= 3:
            break
        if has_ancestor(node, "preproc_if", "preproc_ifdef"):
            continue
        replacement = split_if_else(node, src_bytes)
        if replacement:
            reps.append((node.start_byte, node.end_byte, replacement.encode()))
            split_done += 1

    # 2. Opaque predicate in while loop conditions: up to 2
    loop_done = 0
    for node in find_nodes(body, "while_statement"):
        if loop_done >= 2:
            break
        if has_ancestor(node, "preproc_if", "preproc_ifdef"):
            continue
        replacement = patch_while_condition(node, src_bytes, int_vars)
        if replacement:
            reps.append((node.start_byte, node.end_byte, replacement.encode()))
            loop_done += 1

    # 3. Wrap every 3rd call expression statement: up to 3
    call_done = 0
    for i, node in enumerate(find_nodes(body, "expression_statement")):
        if call_done >= 3:
            break
        if i % 3 != 0:
            continue
        replacement = wrap_expr_stmt_with_predicate(node, src_bytes, int_vars)
        if replacement:
            reps.append((node.start_byte, node.end_byte, replacement.encode()))
            call_done += 1

    # 4. Block flattening on safe sequential statements (one instance per function)
    direct_stmts = [
        c for c in body.named_children
        if c.type in ("expression_statement", "declaration")
        and not has_ancestor(c, "preproc_if", "preproc_ifdef")
    ]
    flat = flatten_sequential_block(direct_stmts, src_bytes)
    if flat:
        reps.append(flat)

    # 5. Fallback: if nothing applied, insert at least one structural opaque predicate
    if not reps:
        stmts = [c for c in body.named_children if c.type in STMT_TYPES]
        if stmts:
            p = _fresh("p")
            insert = f"int {p} = (0 == 0);\n    if ({p} && ({p} == 1)) {{ {p} = {p}; }}\n    "
            reps.append((stmts[0].start_byte, stmts[0].start_byte, insert.encode()))

    return apply_replacements(src_bytes, reps).decode("utf-8", errors="replace")


# ===========================================================================
# OBFUSCATION TYPE 5: DATA-DEPENDENCY PERTURBATION
# Insert reachable no-op assignments (var = var;) for existing local variables
# to create spurious read-after-write edges in the PDG/data-dependency graph.
# The inserted statements are semantically inert but appear as real assignments
# to static analysis tools and models that consume PDG features.
# ===========================================================================

def obf_datadep(src: str) -> str:
    """Insert no-op self-assignments to introduce spurious PDG edges."""
    _reset_ctr()
    src_bytes = src.encode("utf-8", errors="replace")
    root = parse_src(src)

    func_def = get_function_def(root)
    if not func_def:
        return src
    body = get_body(func_def)
    if not body:
        return src

    # Use only function parameters — guaranteed in scope at every insertion point
    candidates = find_int_params(func_def, src_bytes)[:6]
    if not candidates:
        return src

    # Insert self-assignments at statement insertion points
    positions = collect_stmt_insertion_points(body, src_bytes)
    if not positions:
        return src

    # Choose evenly-spaced positions and cycle through candidate variables
    n = len(positions)
    step = max(1, n // (len(candidates) + 1))
    chosen = positions[::step][:len(candidates)]

    reps: List[Replacement] = []
    for i, pos in enumerate(chosen):
        var  = candidates[i % len(candidates)]
        tag  = _fresh("dd")
        text = f"int {tag} = {var}; (void){tag};\n    "
        reps.append((pos, pos, text.encode()))

    return apply_replacements(src_bytes, reps).decode("utf-8", errors="replace")


# ===========================================================================
# File-level driver
# ===========================================================================

OBF_FN = {
    "identifier":   obf_identifier,
    "vocab_shift":  obf_vocab_shift,
    "benign_token": obf_benign_token,
    "deadcode":     obf_deadcode,
    "controlflow":  obf_controlflow,
    "tempvar":      obf_tempvar,
    "datadep":      obf_datadep,
}


def process_file(src_path: Path, dst_path: Path, obf_type: str) -> dict:
    try:
        src = src_path.read_text(encoding="utf-8", errors="replace")
        result = OBF_FN[obf_type](src)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(result, encoding="utf-8")
        return {"status": "ok", "changed": result != src}
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc(),
        }


def main():
    ap = argparse.ArgumentParser(description="Obfuscation transformer for devign_full")
    ap.add_argument("--obf", required=True, choices=["identifier", "vocab_shift", "benign_token",
                                                    "deadcode", "controlflow", "tempvar", "datadep"])
    ap.add_argument(
        "--input", type=Path, default=None,
        help="Input directory (default: devign_full/originals/)",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: devign_full/obf_<type>/)",
    )
    ap.add_argument("--test", type=int, default=None, metavar="N",
                    help="Dry-run: process only first N files, print diffs to stdout")
    args = ap.parse_args()

    devign_full = Path(__file__).resolve().parent
    input_dir = args.input or devign_full / "originals"
    output_dir = args.output or devign_full / f"obf_{args.obf}"

    files = sorted(input_dir.glob("*.c"))
    if args.test:
        files = files[:args.test]

    total = len(files)
    ok = changed = 0
    errors = []

    for i, f in enumerate(files):
        dst = output_dir / f.name
        res = process_file(f, dst, args.obf)

        if res["status"] == "ok":
            ok += 1
            if res["changed"]:
                changed += 1
            if args.test and res["changed"]:
                print(f"\n{'='*60}")
                print(f"FILE: {f.name}")
                print(f"{'='*60}")
                print(dst.read_text(encoding="utf-8", errors="replace")[:2000])
        else:
            errors.append({"file": f.name, "error": res["error"]})
            if args.test:
                print(f"ERROR {f.name}: {res['error']}", file=sys.stderr)

        if not args.test and ((i + 1) % 1000 == 0 or (i + 1) == total):
            print(f"  [{i+1}/{total}] ok={ok} changed={changed} errors={len(errors)}")

    log_path = output_dir / "_obf_log.json"
    with open(log_path, "w") as lf:
        json.dump(
            {
                "obf_type": args.obf,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "total": total,
                "ok": ok,
                "changed": changed,
                "error_count": len(errors),
                "errors": errors[:100],
            },
            lf,
            indent=2,
        )

    print(f"\nDone: {ok}/{total} ok | {changed} transformed | {len(errors)} errors")
    print(f"Log → {log_path}")


if __name__ == "__main__":
    main()
