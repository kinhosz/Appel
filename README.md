# Appel
Computer Vision with Ray Tracing

<!-- Dont modify this line!!! -->
**Current Version**: 0.16.0

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
 Will compile & run all tests inside `tests/` directory.

```sh
make unit FNAME=tests/geometry/vetor.cpp
```
Will compile & run the specific file.

### Dependencies
Cuda toolkit
```
https://linuxhint.com/install-cuda-on-ubuntu-22-04-lts/
https://developer.nvidia.com/cuda-downloads
```

```make
make install
```

Will Install:
* [`SFML v2.6.0`](https://github.com/SFML/SFML): For image processing

**BREAKING CHANGE**: Only Linux suport!

### Project Showcase

- Scenes generated without **Phong shading**:
    - Scenes with only a `TriangularMesh` object
    ![img](/assets/outputs/project_v0/version_00/image_04.png)
    ![img](/assets/outputs/project_v0/version_00/image_03.png)
    
    - Scenes with a `Plane` and `TriangularMesh` objects
    ![img](/assets/outputs/project_v0/version_01/image_05.png)
    ![img](/assets/outputs/project_v0/version_01/image_03.png)

    - Scenes with a `Plane`, `TriangularMesh` and `Sphere` objects
    ![img](/assets/outputs/project_v0/version_02/image_02.png)
    ![img](/assets/outputs/project_v0/version_02/image_01.png)

- Scenes generated with **Phong shading**:
    - Scenes with one `Light` and three objects (`Plane`, `TriangularMesh` and `Sphere`)
    ![img](/assets/outputs/project_v1/version_01/image_01.png)
    ![img](/assets/outputs/project_v1/version_01/image_03.png)

    - Scenes with two `Light`s and three objects (`Plane`, `TriangularMesh` and `Sphere`)
    ![img](/assets/outputs/project_v1/version_02/image_02.png)
    ![img](/assets/outputs/project_v1/version_02/image_04.png)

    - Scenes with three `Light`s and three objects (`Plane`, `TriangularMesh` and `Sphere`)
    ![img](/assets/outputs/project_v1/version_03/image_03.png)
    ![img](/assets/outputs/project_v1/version_03/image_05.png)

- Scene with reflex and refraction:

![img](/assets/outputs/project_v2/version_03/image_00.png)


- A Human Face:

![img](/assets/outputs/view/humanFace.png)
