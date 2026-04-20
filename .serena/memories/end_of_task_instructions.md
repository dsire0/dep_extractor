# End of Task Instructions
1. **Tests & Safety**: Verify all code locally by running `.\run_tests.ps1`.
2. **Atomic Commits**: Execute the `/step-by-step-commits` workflow. Provide an Educational Command Summary explaining commands to View, Undo, and Push the changes.
3. **Beads tracking**: Close the related beads issue using `bd close <id> --reason "..."`.
4. **Pushing code**: Run `git pull --rebase && bd dolt push && git push`. Ensure Beads and Git are synchronized and pushed to origin.
5. **Memory Persistence**: If substantial architectural decisions were made or a specific failure resulted in a rollback, issue a `serena_write_memory` call to harden the project's knowledge base.