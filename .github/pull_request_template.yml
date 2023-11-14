name: Pull request
description: Create a new pull request.
body:
  - type: markdown
    attributes:
      value: |
        Thank you for your contribution to samurai!
  - type: checkboxes
    id: checklist
    attributes:
      label: Requirements
      description: Please check the following before submitting your PR
      options:
        - label: I have installed [pre-commit](https://pre-commit.com/) locally and use it to validate my commits.
        - label: |
            The PR title follows the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) convention.
            Available tags: 'build', 'chore', 'ci', 'docs', 'feat', 'fix', 'perf', 'refactor', 'revert', 'style', 'test'
        - label: This new PR is documented.
        - label: This new PR is tested.
  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear and concise description of what you have done in this PR.
    validations:
      required: true
  - type: textarea
    id: issue
    attributes:
      label: Related issue
      description: List the issues solved by this PR, if any.
  - type: textarea
    id: test
    attributes:
      label: How has this been tested?
      description: Give the list of files used to test this new implementation.
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this PR, you agree to follow our [Code of Conduct](https://github.com/hpc-maths/samurai/blob/master/docs/CODE_OF_CONDUCT.md)
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
