# xpu_cada — m100 GPU-kernel performance overrides

This directory holds m100 (xtrans / xpu_cada) hand-tuned implementations that
**replace** selected kernels under `paddle/phi/kernels/gpu/` (and its
gpu-flavored sibling directories) at compile time.

## How it works

When the build is configured with `-DWITH_XPU_CADA=ON`, `paddle/phi/kernels/CMakeLists.txt`:

1. Globs `xpu_cada/<subdir>/*.cu(.cc)`.
2. For every override file, removes the same-basename file from the original
   `gpu/` (or the corresponding `gpudnn/`, `sparse/gpu/`, ...) list.
3. Appends the override file to the same source list, so it goes through
   `collect_srcs()` and `kernel_declare()` exactly like a stock GPU kernel.

The override compiles into `phi_gpu`, registers under `Backend::GPU`, and is
picked up by the normal kernel dispatch path. No new backend is introduced.

## Authoring rules

- Same **basename** as the file you want to replace.
- Same kernel name and `PD_REGISTER_KERNEL(<name>, GPU, ALL_LAYOUT, ...)`
  signature as the original (otherwise dispatch will not find your kernel,
  or you will silently lose dtype coverage).
- Implement the full set of dtypes the original kernel registers — partial
  override is **not** supported (the whole TU replaces the whole TU).
- Keep host-visible headers (`paddle/phi/kernels/xxx_kernel.h`) unchanged;
  only the implementation `.cu`/`.cu.cc` lives here.

## Layout (mirror gpu-side tree)

| original path                          | put override at                         |
| -------------------------------------- | --------------------------------------- |
| `kernels/gpu/foo_kernel.cu`            | `kernels/xpu_cada/foo_kernel.cu`        |
| `kernels/gpudnn/bar.cu`                | `kernels/xpu_cada/gpudnn/bar.cu`        |
| `kernels/sparse/gpu/baz.cu`            | `kernels/xpu_cada/sparse/gpu/baz.cu`    |
| `kernels/legacy/gpu/qux.cu`            | `kernels/xpu_cada/legacy/gpu/qux.cu`    |
| `kernels/fusion/gpu/quux.cu`           | `kernels/xpu_cada/fusion/gpu/quux.cu`   |

If you need to override a directory that is not yet wired, add a single
`_xpu_cada_override(...)` line in `paddle/phi/kernels/CMakeLists.txt`.

## Verification

At configure time you should see lines like:

```
-- [xpu_cada] override gpu/matmul_kernel.cu -> xpu_cada/matmul_kernel.cu
```

Build artefact check:

```
nm libphi_gpu.a | c++filt | grep MatmulKernel
```

The symbol's containing TU should be the `xpu_cada/...` path, not `gpu/...`.
