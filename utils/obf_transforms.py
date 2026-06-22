"""Shared obfuscation transforms for devign_full (same logic as pilot500)."""
import re


C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
    "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct",
    "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "_Bool",
    "_Complex", "_Imaginary",
}

DECL_PREFIX = {
    "auto", "char", "const", "double", "enum", "extern", "float", "int", "long",
    "register", "restrict", "short", "signed", "static", "struct", "typedef", "union",
    "unsigned", "void", "volatile", "_Bool", "size_t", "ssize_t",
}

SAFE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_param_names(code: str):
    names = []
    lbrace = code.find("{")
    if lbrace == -1:
        return names
    header = code[:lbrace]
    close = header.rfind(")")
    if close == -1:
        return names
    depth = 0
    open_paren = -1
    for i in range(close, -1, -1):
        ch = header[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                open_paren = i
                break
    if open_paren == -1:
        return names
    params = header[open_paren + 1:close].strip()
    if not params or params == "void":
        return names

    chunks = []
    cur = []
    depth = 0
    for ch in params:
        if ch == "," and depth == 0:
            chunks.append("".join(cur))
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        cur.append(ch)
    if cur:
        chunks.append("".join(cur))

    for chunk in chunks:
        ids = SAFE_IDENT.findall(chunk)
        if not ids:
            continue
        cand = ids[-1]
        if cand in C_KEYWORDS:
            continue
        names.append(cand)
    return names


def extract_local_names(code: str):
    names = set()
    for m in re.finditer(r"for\s*\(\s*([^;]+);", code):
        init = m.group(1).strip()
        toks = SAFE_IDENT.findall(init)
        if not toks:
            continue
        if toks[0] not in DECL_PREFIX and not init.startswith((
            "struct ", "enum ", "union ",
        )):
            continue
        for v in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?=\s*(?:=|,|\)|\[))", init
        ):
            name = v.group(1)
            if name not in C_KEYWORDS:
                names.add(name)

    body_start = code.find("{")
    body_end = code.rfind("}")
    if body_start == -1 or body_end == -1 or body_end <= body_start:
        return names
    body = code[body_start + 1 : body_end]
    for stmt in re.finditer(r"([^;{}]+);", body):
        s = stmt.group(1).strip()
        if not s:
            continue
        first = SAFE_IDENT.match(s)
        if not first:
            continue
        first_tok = first.group(0)
        if first_tok not in DECL_PREFIX and not s.startswith(("struct ", "enum ", "union ")):
            continue
        if re.search(r"\)\s*$", s) and "(" in s and "=" not in s:
            continue
        for v in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?=\s*(?:=|,|;|\[))", s
        ):
            name = v.group(1)
            if name not in C_KEYWORDS:
                names.add(name)
    return names


def build_mapping(code: str):
    ordered = []
    seen = set()
    for n in extract_param_names(code):
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    for n in sorted(extract_local_names(code)):
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return {name: f"v{i+1}" for i, name in enumerate(ordered)}


def next_nonspace(s: str, i: int):
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return s[i] if i < n else ""


def prev_nonspace(s: str, i: int):
    i -= 1
    while i >= 0 and s[i].isspace():
        i -= 1
    return s[i] if i >= 0 else ""


def transform_identifiers(code: str, mapping):
    out = []
    i = 0
    n = len(code)
    in_line_comment = False
    in_block_comment = False
    in_str = False
    in_char = False
    esc = False

    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

        if in_line_comment:
            out.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            out.append(ch)
            if ch == "*" and nxt == "/":
                out.append(nxt)
                i += 2
                in_block_comment = False
            else:
                i += 1
            continue
        if in_str:
            out.append(ch)
            if not esc and ch == '"':
                in_str = False
            esc = not esc and ch == "\\"
            if ch != "\\":
                esc = False
            i += 1
            continue
        if in_char:
            out.append(ch)
            if not esc and ch == "'":
                in_char = False
            esc = not esc and ch == "\\"
            if ch != "\\":
                esc = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            out.append(ch)
            out.append(nxt)
            i += 2
            in_line_comment = True
            continue
        if ch == "/" and nxt == "*":
            out.append(ch)
            out.append(nxt)
            i += 2
            in_block_comment = True
            continue
        if ch == '"':
            out.append(ch)
            i += 1
            in_str = True
            esc = False
            continue
        if ch == "'":
            out.append(ch)
            i += 1
            in_char = True
            esc = False
            continue

        if ch == "#" and (i == 0 or code[i - 1] == "\n"):
            j = i
            while j < n and code[j] != "\n":
                out.append(code[j])
                j += 1
            if j < n:
                out.append("\n")
                j += 1
            i = j
            continue

        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] == "_"):
                j += 1
            ident = code[i:j]
            prev = prev_nonspace(code, i)
            nn = next_nonspace(code, j)

            replace = ident in mapping
            if prev == ".":
                replace = False
            if prev == ">" and i >= 2 and code[i - 2] == "-":
                replace = False
            if nn == "(":
                replace = False
            out.append(mapping[ident] if replace else ident)
            i = j
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def detect_indent(code: str, brace_idx: int) -> str:
    after = code[brace_idx + 1 :]
    for line in after.splitlines():
        stripped = line.lstrip(" \t")
        if stripped:
            return line[: len(line) - len(stripped)]
    return "    "


def make_deadcode_block(base_indent: str, tag: int) -> str:
    i1 = base_indent
    i2 = base_indent + "    "
    i3 = base_indent + "        "
    var_flag = f"__dc_flag_{tag}"
    var_tmp = f"__dc_tmp_{tag}"
    return (
        f"\n{i1}int {var_flag} = 0;\n"
        f"{i1}if ({var_flag} == 1) {{\n"
        f"{i2}int {var_tmp} = {tag} + 7;\n"
        f"{i2}{var_tmp} = {var_tmp} * 3;\n"
        f"{i2}for (int __dc_i_{tag} = 0; __dc_i_{tag} < 0; ++__dc_i_{tag}) {{\n"
        f"{i3}{var_tmp} += __dc_i_{tag};\n"
        f"{i2}}}\n"
        f"{i1}}}\n"
    )


def insert_deadcode(code: str, tag: int) -> str:
    brace = code.find("{")
    if brace == -1:
        return code
    indent = detect_indent(code, brace)
    block = make_deadcode_block(indent, tag)
    return code[: brace + 1] + block + code[brace + 1 :]


def indent_block(text: str, indent: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        if line.strip():
            out.append(indent + line)
        else:
            out.append(line)
    return "".join(out)


def rewrite_controlflow(code: str, tag: int) -> str:
    lbrace = code.find("{")
    rbrace = code.rfind("}")
    if lbrace == -1:
        return code

    if rbrace == -1 or rbrace <= lbrace:
        base_indent = detect_indent(code, lbrace)
        fallback = (
            "\n"
            f"{base_indent}int __cf_fallback_{tag} = 1;\n"
            f"{base_indent}if (__cf_fallback_{tag}) {{\n"
            f"{base_indent}    __cf_fallback_{tag} = __cf_fallback_{tag};\n"
            f"{base_indent}}}\n"
        )
        return code[: lbrace + 1] + fallback + code[lbrace + 1 :]

    prefix = code[: lbrace + 1]
    body = code[lbrace + 1 : rbrace]
    suffix = code[rbrace:]

    base_indent = detect_indent(code, lbrace)
    inner_indent = base_indent + "    "

    cond = f"((__cf_gate_{tag} ^ __cf_gate_{tag}) == 0)"
    transformed = (
        "\n"
        f"{base_indent}int __cf_gate_{tag} = {tag};\n"
        f"{base_indent}if {cond} {{\n"
        f"{indent_block(body, inner_indent)}"
        f"{base_indent}}} else {{\n"
        f"{inner_indent}/* unreachable branch for control-flow obfuscation */\n"
        f"{base_indent}}}\n"
    )

    return prefix + transformed + suffix


def numeric_tag_from_filename(stem: str) -> int:
    return int(stem.split("_", 1)[0])
