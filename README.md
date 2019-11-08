# Cybathlon 2020 UniVie

### Repository

This is a monorepository project to store all projects within the Cybathlon 2020 project.

To avoid having cross-platform code-base create a folder for each project in root directory.

### SSH keys

For fluently committing to repository ssh-keys are very usefull.
You can configure git repository to use ssh-keys so you don't have to reenter your passwords over and over.

IF an ssh-key is generated, add them to 
https://github.com/settings/keys

Otherwise you will need to generate one.
I should try really hard to post instructions equally good as github help-page. 
Try to follow [instructions here](https://help.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) 
  

### Using git and the repository

As the power of git lies partly in its branching model, it is highly encouraged to use different branches for different lines of development.

For each feature developed following is suggested.

1. Create a local feature branch
2. Develop a change small enough to cover the feature
3. Push to remote repository and submit a pull request
4. Possibly review with a fellow student/collegue and ask for approval
5. Merge to master branch

To avoid branching conflicts or make them less, try to track current version of master by proactively `git fetch` - ing changes from remote repository and `git rebase master` in case of conflicts.

