"""GitHub integration for config repository management."""

import os
from dataclasses import dataclass
from pathlib import Path

from github import Auth, Github
from github.Repository import Repository

from .schema import ConfigIndex, HPCConfig


@dataclass
class RepoConfig:
    """Configuration for the GitHub repository."""

    owner: str
    repo: str
    branch: str = "main"
    configs_path: str = "configs"


class GitHubConfigStore:
    """Manage HPC configurations in a GitHub repository."""

    def __init__(
        self,
        repo_config: RepoConfig,
        token: str | None = None,
    ):
        """Initialize GitHub config store.

        Args:
            repo_config: Repository configuration
            token: GitHub token. If None, uses GITHUB_TOKEN env var.
        """
        self.repo_config = repo_config
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN env var or pass token.")

        auth = Auth.Token(token)
        self._github = Github(auth=auth)
        self._repo: Repository | None = None

    @property
    def repo(self) -> Repository:
        """Get the GitHub repository."""
        if self._repo is None:
            self._repo = self._github.get_repo(
                f"{self.repo_config.owner}/{self.repo_config.repo}"
            )
        return self._repo

    def _config_path(self, software: str, system: str) -> str:
        """Get the path for a config file in the repo."""
        return f"{self.repo_config.configs_path}/{software}/{system}.yaml"

    def get_config(self, software: str, system: str) -> HPCConfig | None:
        """Fetch a configuration from the repository."""
        path = self._config_path(software, system)
        try:
            content = self.repo.get_contents(path, ref=self.repo_config.branch)
            if isinstance(content, list):
                return None
            yaml_str = content.decoded_content.decode("utf-8")
            return HPCConfig.from_yaml(yaml_str)
        except Exception:
            return None

    def list_configs(self) -> ConfigIndex:
        """List all available configurations."""
        index = ConfigIndex()
        try:
            configs_dir = self.repo.get_contents(
                self.repo_config.configs_path, ref=self.repo_config.branch
            )
            if not isinstance(configs_dir, list):
                return index

            for software_dir in configs_dir:
                if software_dir.type != "dir":
                    continue
                software_name = software_dir.name

                software_contents = self.repo.get_contents(
                    software_dir.path, ref=self.repo_config.branch
                )
                if not isinstance(software_contents, list):
                    continue

                for config_file in software_contents:
                    if config_file.name.endswith(".yaml"):
                        system = config_file.name.replace(".yaml", "")
                        index.add_config(
                            HPCConfig(
                                name=software_name,
                                hpc_system=system,
                                endpoint_id="",
                                execution={"function": ""},
                            ),
                            config_file.path,
                        )
        except Exception:
            pass
        return index

    def save_config(
        self,
        config: HPCConfig,
        message: str | None = None,
        branch: str | None = None,
    ) -> str:
        """Save a configuration to the repository.

        Args:
            config: The configuration to save
            message: Commit message
            branch: Branch to commit to (defaults to repo_config.branch)

        Returns:
            URL of the created/updated file
        """
        path = self._config_path(config.name, config.hpc_system)
        yaml_content = config.to_yaml()
        branch = branch or self.repo_config.branch
        message = message or f"Add config for {config.name} on {config.hpc_system}"

        try:
            # Check if file exists
            existing = self.repo.get_contents(path, ref=branch)
            if isinstance(existing, list):
                raise ValueError(f"Path {path} is a directory")

            # Update existing file
            self.repo.update_file(
                path=path,
                message=message,
                content=yaml_content,
                sha=existing.sha,
                branch=branch,
            )
        except Exception:
            # Create new file
            self.repo.create_file(
                path=path,
                message=message,
                content=yaml_content,
                branch=branch,
            )

        return f"https://github.com/{self.repo_config.owner}/{self.repo_config.repo}/blob/{branch}/{path}"

    def create_config_pr(
        self,
        config: HPCConfig,
        title: str | None = None,
        body: str | None = None,
    ) -> str:
        """Create a pull request with a new configuration.

        Args:
            config: The configuration to add
            title: PR title
            body: PR body/description

        Returns:
            URL of the created PR
        """
        # Create a new branch
        branch_name = f"config/{config.name}-{config.hpc_system}"
        base_ref = self.repo.get_branch(self.repo_config.branch)

        # Check if branch exists, if so make unique
        try:
            self.repo.get_branch(branch_name)
            import time
            branch_name = f"{branch_name}-{int(time.time())}"
        except Exception:
            pass

        self.repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=base_ref.commit.sha,
        )

        # Save config to new branch
        self.save_config(
            config,
            message=f"Add config for {config.name} on {config.hpc_system}",
            branch=branch_name,
        )

        # Create PR
        title = title or f"Add {config.name} configuration for {config.hpc_system}"
        body = body or self._generate_pr_body(config)

        pr = self.repo.create_pull(
            title=title,
            body=body,
            head=branch_name,
            base=self.repo_config.branch,
        )

        return pr.html_url

    def _generate_pr_body(self, config: HPCConfig) -> str:
        """Generate a PR body from a config."""
        lines = [
            f"## Configuration for {config.name} on {config.hpc_system}",
            "",
        ]

        if config.description:
            lines.extend([config.description, ""])

        if config.version:
            lines.append(f"**Version:** {config.version}")

        if config.environment.modules:
            lines.append(f"**Modules:** {', '.join(config.environment.modules)}")

        if config.environment.conda_env:
            lines.append(f"**Conda env:** {config.environment.conda_env}")

        lines.extend([
            "",
            "### Discovery Log",
            f"- **Date:** {config.discovery_log.date}",
            f"- **Attempts:** {config.discovery_log.attempts}",
        ])

        if config.discovery_log.docs_consulted:
            lines.append("- **Docs consulted:**")
            for doc in config.discovery_log.docs_consulted:
                lines.append(f"  - {doc}")

        if config.discovery_log.notes:
            lines.extend(["", f"**Notes:** {config.discovery_log.notes}"])

        return "\n".join(lines)

    def search_configs(self, query: str) -> list[tuple[str, str]]:
        """Search for configs matching a query.

        Returns:
            List of (software, system) tuples
        """
        results = []
        index = self.list_configs()
        query_lower = query.lower()

        for software in index.list_software():
            if query_lower in software.lower():
                for system in index.list_systems(software):
                    results.append((software, system))
            else:
                for system in index.list_systems(software):
                    if query_lower in system.lower():
                        results.append((software, system))

        return results
