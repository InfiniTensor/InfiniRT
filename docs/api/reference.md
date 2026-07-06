# API Reference

InfiniRT keeps user-facing documentation in Markdown and can generate a C++ API
reference with Doxygen.

## Generate Reference HTML

Configure a build first so that generated public headers exist:

```bash
cmake -S . -B build -DWITH_CPU=ON
```

Then generate it through CMake:

```bash
cmake -S . -B build -DINFINI_RT_BUILD_DOCS=ON
cmake --build build --target infinirt_docs
```

The generated HTML is written under:

```text
build/docs/reference/html
```

## Preview the Rendered Structure

Serve the generated HTML directory with a local static file server:

```bash
python3 -m http.server 8000 --directory build/docs/reference/html
```

Then open:

```text
http://localhost:8000/
```

The Doxygen page includes a left-side tree view and search box, which are the
recommended way to inspect the rendered API structure.

## Reference Scope

The Doxygen configuration is intentionally scoped to the public entry header,
core public headers, and generated dispatching runtime declarations.

The following are not intended as the primary user documentation surface:

- `infini/rt/detail/*`
- backend implementation detail copied from `src/native/*`
- generated implementation source under `generated/src`

Use [Compatibility](../compatibility.md) for the supported API boundary.
