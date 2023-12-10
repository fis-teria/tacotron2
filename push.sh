#!/bin/bash -e
COMMIT_STEP=1  # 何コミットずつpushするか。1でよい。時間がかかるようなら、100とかを試してもよい。
START_COMMIT=1 # 一番最初は何コミット目から始めるか。通常1でよい。

git log --pretty=%H | tac > commits
COMMIT_COUNT=$(wc -l commits | cut -d' ' -f1)

for i in `seq $START_COMMIT $COMMIT_STEP $COMMIT_COUNT`; do
	echo ====== $i
	COMMIT=$(head -$i commits | tail -1)
	git tag -d foo || true
	git tag foo $COMMIT
	git push -f origin foo
done

git tag -d foo
git push origin HEAD
git push --mirror
git lfs push --all  # 念のため

