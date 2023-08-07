# Appel
Computer Vision with Ray Tracing

<!-- Dont modify this line!!! -->
**Current Version**: 0.5.0

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
 