#!/usr/bin/env bash

set -euo pipefail

contains_dependency_all=true

while read -r package version; do
    # python is declared via requires-python; pip and hatchling are only needed for the
    # editable install in the pixi environments, not at runtime
    if [[ $package == "python" || $package == "pip" || $package == "hatchling" ]]; then
        continue
    fi

    dependency="${package} ${version}"
    contains_dependency=$(yq -r ".project.dependencies | map(. == \"${dependency}\") | any" pyproject.toml)
    if [[ $contains_dependency == "false" ]]; then
        echo "${dependency} not found in pyproject.toml"
        contains_dependency_all=false
    fi
done < <(yq -r '.dependencies | to_entries | .[] | "\(.key) \(.value)"' pixi.toml)

if [[ $contains_dependency_all == "false" ]]; then
    exit 1
fi
