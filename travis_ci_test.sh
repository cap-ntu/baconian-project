#!/usr/bin/env bash
git checkout -b travis_ci_test
git push -u origin travis_ci_test
git push -u public travis_ci_test
