# Contributing guidelines

If you have improvements to the xFasterTransformer code, please send us your pull
requests! To get started, see the GitHub
[howto](https://help.github.com/en/articles/about-pull-requests).

You can:

- Submit your changes directly with a
  [pull request](https://github.com/intel/xFasterTransformer/pulls)
- Log a bug or feedback with an [issue](https://github.com/intel/xFasterTransformer/issues)

**See also:** [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## Pull request checklist

Before sending your pull requests, make sure that you have followed this list:

* Check the [library functionality guidelines](CONTRIBUTING.md#library-functionality-guidelines).


* Ensure that the changes are consistent with the
  [code contribution guidelines](CONTRIBUTING.md#code-contribution-guidelines)
  and [coding standards](CONTRIBUTING.md#coding-standards).

* Check that [unit tests](CONTRIBUTING.md#unit-tests) pass.

## Library functionality guidelines

xFasterTransformer focuses on functionality that satisfies all of the following
criteria:

1. *Performance*: the functionality has material impact on a workload level.
   In other words, this means that for a optimization it should be
   demonstrated that it brings visible performance improvement to some
   workload.

2. *Generality*: when introducing new foundational features, their API should
    be sufficiently versatile and user-friendly to facilitate integration into
    other frameworks.

3. *Complexity*: it is not trivial to implement the functionality directly in
   a LLM application.

## Code contribution guidelines

When submitting your contribution, please make sure that it is:

* *Tested*: xFasterTransformer uses gtests for lightweight functional testing. Please make 
  your contribution is fully tested by unit tested.

* *Documented*: Please add essential inline comments to aid others in comprehending the 
  code. When necessary, include appropriate documentation explanations.

All code in xFasterTransformer gets promoted to product branches (`main` and `rls-`) 
only through GitHub pull requests. Requirements for promotion:

- The request is reviewed and approved by maintainers for all affected
  components.
- All discussions in the pull request are resolved.
- Continuous integration pipeline passed without errors.
- Promotion to release (`rls-`) branches can be done only by maintainers
  (enforced by GitHub)
- The pull request author is responsible for collecting all the necessary
  approvals, rebasing of the changes, and resolving the discussions.

To simplify the work of reviewers, make sure that the commits in the pull
request adhere to the following requirements:

- Commit message should be fit into 50 (at most 72) characters and have the
  imperative mood.
- Commit message should follow the format:
  `[Scope][Scope: ..] <short description>`
  Scope examples:
  * Top level: `Build`, `API`, `Doc`, `Tests`, `Common`, `Models`, `Kernels`
  * Second level: `BF16`, `Layers`, `Utils`, `Searchers`
  * Example commit message:
~~~git
[Kernels][BF16]: Add AMX format BA16a64b2a.
~~~

- Commit body should also fit 72 characters. Think of it as a standard e-mail
  body or a markdown document in terms of styling - write sentences from the
  very left border keeping capital letters and punctuation in place.
- xFasterTransformer branches maintain linear history. Rebase the changes on top of target
  branch before creating a pull request. Rebase again after resolving all the
  discussions, as well as in case of merge conflicts.
- Use `git add -p`  and `git rebase -i` liberally to split unrelated changes
  into multiple self-contained commits. This is a courtesy to reviewers: smaller
  commits are easier to comprehend. It also helps with bisecting in the future.
  Of course judgement needs to be applied whether to split changes or not. For
  example, split code cleanup and the actual fix into two separate patches.

## Coding Standards

Contributions to xFasterTransformer must follow the [Coding Standards](CODING_STANDARDS.md)
in order to simplify development and review processes. The general principle is
to follow the style of existing/surrounding code.

The Coding Standards are subject to change and contributions to the Coding
Standards are welcome.

If you wish to propose changes to the Coding Standards (including `clang-format`
 options), please submit the proposal via an pull request. The proposal should 
 contain the following information:
* *Motivation*: Why should the proposed standard be introduced and applied?
* *Enforcement*: Can the proposed standard be applied via an automated process
  or other practical means?
* *Example*: What does the code base look like with the proposed standard
  applied?

## Unit tests

xFasterTransformer uses gtests for lightweight functional testing.

Be sure to extend the existing tests when fixing an issue.