# Rodent-experiments
The "rodent-experiments" project is a python-based data science project dealing with
object-detection and classification of camera trap images of small animals: rodents, snakes, some birds, both for RGB only and RGB-T.

## Tech stack
- python >= 3.13
- DVC for data management and pipeline definition
- pandas and matplotlib for data analysis
- ultralytics with yolo26 for general object detection and image classification, and model fine tuning
- pytorch for custom model building
- yaml for config files
- jupyter for notebooks
- speciesnet for camera-trap specific object detection
- inaturalist and gbif for image acquisition
- vllm and ollama for image filtering based on VLLM models
- pytest as general testing framework and 'Hypothesis' for proprety-based testing

## Structure
- dvc.yaml: Contains pipeline definition
- scripts: Contains excecutable scripts. These make up the steps of the dvc pipeline
- datasets: Contains image data, raw and processed
- configs: Contains yaml files that parameterize pipeline steps
- src/smartrodent: Contains the library code

## Usage
This project uses DVC as its core driver.
- A DVC pipeline is defined in ./dvc.yaml which can be used to find out the steps
- Always run from an activated virtual environment with this project installed in it in editable mode (because this is still WIP)
- Run a step using: `dvc repor 'step_name'`
- Run tests with `pytest --cov=smartrodent --cov-report=term-missing` to run the tests and get line coverage report and `pytest --cov=smartrodent --cov-report=term-missing --cov-branch` to get branch coverage. Both are important

## Restrictions
- do not commit anything to git unless told so explicitly
- never commit secrets into git, e.g., API keys
- never edit code without asking first and giving a clear step-by-step plan first
- never modify the data in the ./datasets directory unless explicitly told to do so. It contains the data of the project in various stages or. processing and must hence be held sacred! If told to modify it, raise the concern that this is normally forbidden first, plan out a step-by-step workflow and ask for explicit permission to execute it first.
- config files are treated as ephemeral and do not need tests

## Coding guidelines
- Adhere to SOLID principles
    - single responsibility principle: One code unit (function, class, module) should have one single purpose
    - open-closed principle: Code should be open to extension, but closed to modification, i.e., we should not have to modify existing code extensively to extend the functionality of the system
    - liskov substitution: derived- or child classes must be able to replace their parent classes, i.e., child classes should extend the parent classes without breaking the behavioral contract
    - interface segregation: A client should not be forced to depend on parts of an interface that is not relevant to it
    - dependency inversion: Higher-level elements of the code should not depend on lower-level elements of the code, both should depend on abstractions that define behavior, not restrict implementation.
- keep control flow and data flow as flat as possible. Go with the minimum amount of hierarchical dept that solves the problem at hand.
- avoid module-level variables
- use classes as soon as data and functions work together to create a functionality
- avoid deeply nested branching
- branches should be self-explanatory
- make branches explicit. Do not rely on handling a case by leaving it as the last one that's missing, rather use explicit if-elif-else blocks
- when writing tests, test behavior and state, not it's implementation details
- keep the hot path of a function clean: configuration, parameter handling and preparation should happen on as high a level as possible
- handle edge cases explicitly. Do not rely on silent defaults that can alter behavior without the user knowing
- exceptions should be handled explicitly and early when they are expected or anticipated. In cases where unexpected/unanticipated things happen, make them bubble up to the highest level and handle them there:
```python
try:
    func()
except AnticipatedException1 as e1:  # might happen and we know what to do
    handle_e1()
except AnticipatedException2 as e2:  # might happen and we know what to do
    handle_e2()
except Exception as e:  # unknown exception branch for unexpected stuff
    raise SpecificErrorForThisCase("Specific message here")
```
- no changes to the code are made without a corresponding test of the new or changed funcationality
- property-based testing is a first-class test system
- all public methods and classes should have google style docstrings
- all private methods or classes should have at least freeform docstrings
- code should have explanatory comments that explains decisions taken where multiple alternatives would be possible or where a certain algorithm to solve a problem is first introduced
- adhere to clean code principles with practical application of the principles of "Clean Code" by Robert C. Martin, and "Refactoring" by Martin Fowler, but with relaxed rules on naming conventions and function length. The code should be readable and understandable, but not necessarily follow strict naming conventions or have very short functions. The focus is on clarity and maintainability rather than strict adherence to style guides.