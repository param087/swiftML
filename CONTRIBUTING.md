# Contributors Guideline to swiftML

## Contributing code
All submissions, including submissions by project members, require review. We use GitHub pull requests for this purpose. Consult [GitHub Help](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) for more information on using pull requests.

## Beginner Issues
If you are new to the project and want to get into the code, we recommend picking an issue with the label "good first issue". These issues should only require general programming knowledge and little to none insights into the project.

## Issue Allocation
Each issue someone is currently working on should have an assignee. If you want to contribute to an issue someone else is already working on please make sure to get in contact with that person via slack or github and organize yourself.

If you want to work on an open issue, please post a comment telling that you will work on that issue, we will assign you as the assignee then.

**Caution**: We try our best to keep the assignee up-to-date but as we are all humans with our own schedule delays are possible, so make sure to check the comments once before you start working on an issue even when no one is assigned to it.

## Contribution guidelines and standards

Before sending your pull request for [review](https://github.com/param087/swiftML/pulls), make sure your changes are consistent with the guidelines.

### Testing
* Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.

### Swift coding style
Changes should conform to:

[Google Swift Style Guide](https://google.github.io/swift/)
[Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/)

With the exception that 4-space indendation be used.