# Appel
Computer Vision with Ray Tracing

<!-- Dont modify this line!!! -->
**Current Version**: 0.24.2

> Project for the Graphics Processing discipline of Computer Science course @ CIn/UFPE.
## Contributing guide
To facilitate development, the project will follow the below conventions:
* Task management through GitHub's Kanban, using `issues`;
* `main` branch protected: only changes via pull requests are allowed;
* Pull requests require at least one approval to be merged;
* Maintain the project structure to avoid complications in merges;
* Document everything possible whenever feasible.

### See more

[Project Structure](./docs/project_structure.md)

### Compile
```sh
make tests
```
 Will compile & run all tests inside `tests/` directory using `GPU`.

```sh
make tests EGPU=0
```
 Will compile & run all tests inside `tests/` directory using `CPU` (EGPU = Enable GPU).

```sh
make unit FNAME=tests/geometry/vetor.cpp
```
Will compile & run the specific file using `GPU`.
```sh
make unit FNAME=tests/geometry/vetor.cpp EGPU=0
```
Will compile & run the specific file using `CPU`.

### clear bin
```sh
make clear
```

### Dependencies
Cuda toolkit(optional)
```
https://linuxhint.com/install-cuda-on-ubuntu-22-04-lts/
https://developer.nvidia.com/cuda-downloads
```

```make
make install
```

Will Install:
* [`SFML v2.6.0`](https://github.com/SFML/SFML): For image processing

> Note: On windows, use this compiler: https://github.com/brechtsanders/winlibs_mingw/releases/download/13.1.0-16.0.5-11.0.0-msvcrt-r5/winlibs-x86_64-posix-seh-gcc-13.1.0-mingw-w64msvcrt-11.0.0-r5.7z

**BREAKING CHANGE**: (GPU feature) Only Linux support!

### Project Showcase

- Scenes generated with **Phong shading**:
    - Scenes with one `Light` and three objects (`Plane`, `TriangularMesh` and `Sphere`)
    ![img](/assets/outputs/project_v1/version_01/image_01.png)
    ![img](/assets/outputs/project_v1/version_03/image_05.png)

    - Scene with reflex and refraction:
    ![img](/assets/outputs/project_v2/version_00/image_02.png)
    ![img](/assets/outputs/project_v2/version_02/image_00.png)

    - A Human Face:

    ![img](/assets/outputs/view/humanFace.png)
