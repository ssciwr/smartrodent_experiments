# Rodent-experiments

The "rodent-experiments" project is a python-based data science project dealing with
object-detection and classification of camera trap images of small animals: rodents, snakes, some birds, both for RGB only and RGB-T.

## Tech stack

- python >= 3.12
- DVC for data management and pipeline definition
- pandas and matplotlib for data analysis
- ultralytics with yolo26 for general object detection and image classification, and model fine tuning
- pytorch for custom model building
- yaml for config files
- jupyter for notebooks
- speciesnet for camera-trap specific object detection
- inaturalist and LILA for image acquisition
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

## Role separation

The main agent acts as an orchestrator, not an implementer.

- Delegate implementation, testing and refactoring.
- Delegate verification to independent reviewers.
- Retain responsibility for architectural coherence.
- Resolve conflicts between subagents.
- Escalate specification ambiguities and consequential
  architectural decisions to the human.
- Do not make unapproved changes to the specification.
- Require concise, evidence-based subagent reports.

## Workflow

When implementing features, work like this:

- use the test driven development paradigm
- after the user gives you the requirements, use the following process when tasked with implementation:
    1) plan: Write a plan for implementation including the test list and -functionality (one line per behaviour, Hypothesis candidates marked). Ask the human to approve plan and test list in one stop.
    2) tests and implementation: one implementation subagent writes the approved test list, confirms it is red for the right reason (missing code, not test bugs), then writes the minimal code that turns the tests green: only what the plan and tests require, no speculative helpers or compatibility shims.
    3) review: the reviewer subagent reviews tests and implementation together, at most 4 rounds. After each round the same implementation subagent (resumed, not a new one) fixes all findings. Rounds after the first only review the changes since the previous round and check that the previous findings are resolved. Stop as soon as a round has no blocker or should-fix findings; nits are fixed without another round.
    4) refactor: run the refactorer subagent only when the review flags unneeded complexity; its result gets one review round.
    5) final human review, iterate until the human approves
- the human sees the tests again only with the final result, unless writing or reviewing them raises a specification question
- the full test suite runs once per step, by the subagent doing that step, in the foreground (targeted test files while iterating are fine). It reports the commands and counts. The orchestrator does not re-run it.
- keep context small: delegation prompts name the exact files to read and the scope; subagent reports and orchestrator messages are short and contain only results, decisions and open questions
- the delegation prompt is the handoff: it contains the approved test list, the API signatures the tests target, and the files to create or modify. The subagent reads only those files plus one existing test file for style. It does not read the plans, AGENTS.md or unrelated modules; whatever it needs from them belongs in the prompt.

## Coding guidelines

- Adhere to SOLID principles

  - single responsibility principle: One code unit (function, class, module) should have one single purpose
  - open-closed principle: Code should be open to extension, but closed to modification, i.e., we should not have to modify existing code extensively to extend the functionality of the system
  - liskov substitution: derived- or child classes must be able to replace their parent classes, i.e., child classes should extend the parent classes without breaking the behavioral contract
  - interface segregation: A client should not be forced to depend on parts of an interface that is not relevant to it
  - dependency inversion: Higher-level elements of the code should not depend on lower-level elements of the code, both should depend on abstractions that define behavior, not restrict implementation.

- Use the DRY principle. Avoid repetition.
- Use the KISS principle. Avoid complexity and complicated structures wherever possible, keep things simple.
- keep control flow and data flow as flat as possible. Go with the minimum amount of hierarchical depth that solves the problem at hand.
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