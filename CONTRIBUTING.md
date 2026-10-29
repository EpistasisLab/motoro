# Contributing and releasing

## Branches and pull requests

`develop` is the default branch. It collects completed changes for integration
testing before a release. `main` holds code promoted for release.

| Branch | Start from | PR target | Merge method |
| --- | --- | --- | --- |
| `feat/*`, `fix/*`, `docs/*` | `develop` | `develop` | Squash preferred |
| Release promotion from `develop` | Existing `develop` | `main` | Merge commit |
| `hotfix/*` | `main` | `main` | Merge commit |
| Synchronization from `main` | Existing `main` | `develop` | Merge commit |

Create new work from the latest remote `develop`. Keep unfinished changes on
their feature branches, or behind disabled feature flags, so `develop` remains
suitable for release validation. An already merged PR is complete: follow-up
work belongs on a new branch from `develop` and in a new PR.

Feature PRs describe the problem, resulting behavior, and relevant validation.
Squash merges keep each feature together. Release and synchronization PRs use
merge commits to preserve shared ancestry between the permanent branches.

## Required checks and protection

Both `develop` and `main` require:

- Changes submitted through a PR.
- The GitHub Actions check `Lint, Type Check & Test` to pass against the latest
  target branch.
- Review conversations to be resolved.
- Protection against force pushes and branch deletion, with no bypass actors.

The required reviewer approval count is zero to support solo maintenance.
Review is encouraged, but an approving review from a second account is not a
merge prerequisite. The author remains responsible for inspecting the diff and
validation evidence before merging.

`develop` allows squash and merge commits. `main` allows merge commits only.
Targeting `main` only for release promotion or hotfixes is a workflow convention;
the rulesets do not restrict a PR's source branch.

CI runs for pushes and PRs targeting both permanent branches. Consult
[the CI workflow](.github/workflows/ci.yml) for the commands and blocking status
of each step. Formatting and type checking currently report warnings without
failing the job, so a green check is not proof those steps were clean. Review
their output and account for any findings relevant to the change.

The active GitHub settings are available in the
[develop ruleset](https://github.com/EpistasisLab/motoro/rules/21595294) and
[main ruleset](https://github.com/EpistasisLab/motoro/rules/26162445).
Update this document when changing the workflow or protection policy.

## Validate an integration

1. Implement fixes and regression tests on a feature branch. Run the checks
   relevant to the change and submit the PR to `develop`.
2. After merging, verify CI on the resulting `develop` commit. This checks the
   combined state intended for the next release.
3. Validate affected dependent products against that exact Motoro commit.
   Record the Motoro and product commit IDs, commands, results, and any remaining
   limitations in the product PR or release PR. A moving branch name alone is
   not a reproducible dependency pin.

For the provider integration in ASAREE, exercise its actual service/API handlers
and existing configurations. Cover credential resolution and secrecy, connection
statuses and errors, model discovery and fallback, metadata/default mapping,
inference routing and sampling, and application-owned caching and user isolation.
Use deterministic provider responses for automated integration tests. Smoke-test
the credential-to-model-selection-to-agent-run journey, and use authorized test
credentials for live discovery and small inference requests where applicable.
Keep a record of providers that were mocked versus tested live.

## Release Motoro and update dependent products

1. Confirm CI passes on the intended `develop` commit and dependent-product
   compatibility validation is complete. The release includes the combined
   changes on `develop`.
2. Open a release PR from `develop` into `main`, for example `Release v0.14.0`.
   Include the proposed version, release notes, validation evidence, and known
   limitations. Complete release review before merging.
3. Merge using a merge commit and verify CI on the resulting `main` commit.
   Treat promotion to `main` as a release decision; check downstream automation
   before merging.
4. Tag the verified `main` commit with the approved semantic version:

   ```bash
   git fetch origin
   git tag -a vX.Y.Z origin/main -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

   Verify `origin/main` is the intended release commit before tagging. The tag
   is the package version: `hatch-vcs` derives it at build time, so there is no
   version string to edit. Tags identify releases and must remain immutable.
5. In ASAREE, replace the temporary Motoro commit pin with the tested release
   tag on a feature branch, regenerate its lockfile using its package manager,
   and rerun validation. Merge that PR into ASAREE's `develop` and verify its CI.
   Promote ASAREE to `main` only after these checks. Motoro releases first.

For an urgent production fix, create `hotfix/*` from `main`, validate it, and
submit its PR to `main`. Release a new patch tag after the merge and successful
CI. Then open a `main` to `develop` synchronization PR and merge it using a merge
commit so future releases retain the fix.

Coding agents follow the branch and PR workflow above. Obtain explicit user
authorization before merging into `main`, pushing release tags, publishing
artifacts, or deploying production. Authorization already given for a specific
action in the session remains valid.
