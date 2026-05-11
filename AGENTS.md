# AGENTS.md

This repository contains a local-first personal lifelog search and question-answering application.

The app ingests locally exported photos and LINE chat histories, extracts timestamps, EXIF GPS, metadata, and message contents, then builds a searchable personal timeline. The final goal is to answer questions such as:

- "What was I doing on 2024-12-24?"
- "When did I go to Shinjuku?"
- "When was the last time I met a specific person?"
- "Summarize my trips from last summer."
- "Find days when I ate ramen."

This app handles highly private personal data, including photos, locations, timestamps, and private messages. Privacy and local-only execution are the most important requirements.

---

## 1. Repository path

The expected project path is:

```bash
~/MyApplication/personal_lifelog_rag/
```


---

## 21. Conda environment policy

This project must be developed inside a dedicated Conda environment.

Do not install dependencies into the `base` environment.

The default Conda environment name is:

```bash
personal_lifelog_rag
```

キリのいいところでGit更新はしてください。

---

## 22. Portfolio HTML synchronization

When `docs/*.md`, `reports/*.md`, or `README.md` are edited for portfolio,
reporting, or public presentation content, also rebuild the public portfolio
HTML:

```bash
python -m personal_lifelog_rag.app.cli build-portfolio-html --mode public --check-privacy --force
```

Confirm the generated HTML passes:

```bash
python scripts/check_public_portfolio_safety.py reports/portfolio_public.html
```

Do not publish generated HTML unless the privacy check passes and a human has
confirmed it contains no real photos, raw LINE text, exact location
coordinates, personal names, local absolute paths, or private runtime settings.



## Monthly rollout rule

月次展開は必ず1か月単位で行う。一度に全期間を処理しない。

各月の処理順:
1. month-plan
2. month-run --dry-run
3. backup-db
4. month-run --yes
5. month-status
6. 月次QA確認
7. db-check --strict
8. report生成
9. portfolio HTML再生成
10. privacy check
11. docs/monthly_rollout_status.md更新
12. OCR本実行

db-check --strict または privacy check が失敗した場合は、次の月へ進まない。
docs/*.md, reports/*.md, README.md を編集した場合は、必ず portfolio_public.html を再生成する。

---

## New data ingestion refresh rule

新しい写真、LINE履歴、GPS付きmedia、通話ログ、OCR対象画像などを読み込んだ後は、
単にDBへ入れるだけで終わらせない。追加された日付範囲に対して、派生データを安全に
再生成する。

原則:

- 外部API禁止、クラウド送信禁止、モデル自動ダウンロード禁止。
- 実写真、LINE本文、正確なGPS、顔crop、顔embeddingを公開用docs/reports/HTMLへ出さない。
- DBを書き換える前に必ず `backup-db` を実行する。
- 重い処理は必ず `--dry-run` または plan/status で対象件数を確認する。
- 大量の全期間再処理は避け、可能な限り月単位または日付範囲単位で処理する。
- `db-check --strict` が失敗したら次の工程へ進まない。
- UIや別プロセスがDBを掴んでいないか確認してから、embedding/clusteringなどの重いDB更新を行う。

事前確認:

```bash
python -m personal_lifelog_rag.app.cli db-check --strict
python -m personal_lifelog_rag.app.cli month-plan --month YYYY-MM
python -m personal_lifelog_rag.app.cli month-status --month YYYY-MM
ps aux | grep -E "personal_lifelog_rag|gradio|python" | grep -v grep || true
lsof data/db/lifelog.sqlite || true
```

月単位で新規データを処理できる場合は、まず `month-run` を使う:

```bash
python -m personal_lifelog_rag.app.cli month-run \
  --month YYYY-MM \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config private_config/model_runtime.yaml \
  --save-report \
  --dry-run

python -m personal_lifelog_rag.app.cli backup-db --label before_month_run_YYYYMM

python -m personal_lifelog_rag.app.cli month-run \
  --month YYYY-MM \
  --vlm-limit 300 \
  --embedding-limit 300 \
  --config private_config/model_runtime.yaml \
  --save-report \
  --yes
```

月次処理後に必ず確認する:

```bash
python -m personal_lifelog_rag.app.cli month-status --month YYYY-MM
python -m personal_lifelog_rag.app.cli qa "YYYY年M月は何していた？"
python -m personal_lifelog_rag.app.cli db-check --strict
```

### Required derived-data refresh after new media

新しいmediaが入ったら、必要に応じて以下を更新する。

1. VLM画像解析
   - `analyze-images` または `month-run` のVLM stepで更新する。
   - fake VLMやfailed/engine_unavailableは通常検索・event evidenceへ混ぜない。

2. Multimodal embeddings
   - `build-image-embeddings`
   - `build-text-embeddings`
   - または `month-run` のembedding stepで更新する。
   - Qwen3-VL-Embeddingはlocal modelのみ使い、モデル自動ダウンロードは禁止。

3. OCR
   - 全画像ではなく、まず `ocr-priority` で文字が写っていそうな画像を確認する。
   - OCR-onlyでは断定しない。

```bash
python -m personal_lifelog_rag.app.cli ocr-priority \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --limit 50

python -m personal_lifelog_rag.app.cli ocr-images \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --engine tesseract_cli \
  --text-cues-only \
  --limit 50 \
  --skip-existing
```

4. Location / places
   - GPS付きmediaから `location_points` を作る。
   - `place_clusters` を再計算する。
   - `places`辞書を使って `media_places` / `event_places` を再付与する。
   - public reportではexact GPSを絶対に出さない。

```bash
python -m personal_lifelog_rag.app.cli build-location-points \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --dry-run

python -m personal_lifelog_rag.app.cli backup-db --label before_location_refresh_YYYYMMDD

python -m personal_lifelog_rag.app.cli build-location-points \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --yes

python -m personal_lifelog_rag.app.cli cluster-places \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --eps-meters 100 \
  --min-samples 3 \
  --dry-run

python -m personal_lifelog_rag.app.cli assign-places \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --dry-run
```

5. Face detection
   - 新しいmediaにはYuNetを優先する。
   - Haarはfallbackであり、精度確認なしに大量採用しない。
   - 顔cropはlocal private dataとして扱い、公開しない。

```bash
python -m personal_lifelog_rag.app.cli face-diagnostics \
  --config private_config/model_runtime.yaml

python -m personal_lifelog_rag.app.cli face-detect \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --engine opencv_yunet \
  --config private_config/model_runtime.yaml \
  --save-crops \
  --dry-run

python -m personal_lifelog_rag.app.cli backup-db --label before_face_detect_YYYYMMDD

python -m personal_lifelog_rag.app.cli face-detect \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --engine opencv_yunet \
  --config private_config/model_runtime.yaml \
  --save-crops \
  --skip-existing
```

6. Face embeddings
   - YuNet由来かつ `status=success` のface detectionsだけを対象にする。
   - SFace model pathは `private_config/model_runtime.yaml` で指定する。
   - 既存の手動person linkがある場合、face cluster置換は慎重に行う。

```bash
python -m personal_lifelog_rag.app.cli face-embedding-diagnostics \
  --config private_config/model_runtime.yaml

python -m personal_lifelog_rag.app.cli face-embed \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --config private_config/model_runtime.yaml \
  --engine opencv_sface \
  --detections-engine opencv_yunet \
  --status success \
  --only-existing-files \
  --batch-size 500 \
  --dry-run

python -m personal_lifelog_rag.app.cli backup-db --label before_face_embed_YYYYMMDD

python -m personal_lifelog_rag.app.cli face-embed \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --config private_config/model_runtime.yaml \
  --engine opencv_sface \
  --detections-engine opencv_yunet \
  --status success \
  --only-existing-files \
  --batch-size 500 \
  --skip-existing \
  --save-report
```

7. Face clustering
   - 新規顔embeddingを作ったら、クラスタリングは必ずdry-run比較してから実行する。
   - 巨大clusterが出た設定は採用しない。
   - 生成されたface clustersはすべて未確認候補であり、QA/search/reportに使わない。
   - person_face_clustersが存在するscopeを `--replace` する場合は、手動リンクを壊さないよう停止して確認する。

```bash
python -m personal_lifelog_rag.app.cli face-cluster \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --method dbscan_cosine \
  --distance-threshold 0.10 \
  --min-samples 3 \
  --scope yunet_YYYYMMDD_YYYYMMDD \
  --replace \
  --dry-run

python -m personal_lifelog_rag.app.cli backup-db --label before_face_cluster_YYYYMMDD

python -m personal_lifelog_rag.app.cli face-cluster \
  --from YYYY-MM-DD \
  --to YYYY-MM-DD \
  --method dbscan_cosine \
  --distance-threshold 0.10 \
  --min-samples 3 \
  --scope yunet_YYYYMMDD_YYYYMMDD \
  --replace \
  --yes \
  --save-report
```

8. Person / event integration
   - `build-media-people` と `build-event-people` は、手動確認済みperson・face cluster・LINE speaker linkがある場合のみ実行する。
   - 未確認face clusterや未確認personをmedia_people/event_peopleに使わない。

```bash
python -m personal_lifelog_rag.app.cli people-stats
python -m personal_lifelog_rag.app.cli build-media-people --from YYYY-MM-DD --to YYYY-MM-DD --dry-run
python -m personal_lifelog_rag.app.cli build-event-people --from YYYY-MM-DD --to YYYY-MM-DD --dry-run
```

### Final validation after new data refresh

派生データ更新後は必ず以下を実行する。

```bash
python -m personal_lifelog_rag.app.cli db-check --strict
python -m personal_lifelog_rag.app.cli face-embedding-stats --from YYYY-MM-DD --to YYYY-MM-DD
python -m personal_lifelog_rag.app.cli face-cluster-stats
python -m personal_lifelog_rag.app.cli privacy-audit --public
```

公開用docs/reports/READMEを更新した場合、または公開ポートフォリオへ反映する場合:

```bash
python -m personal_lifelog_rag.app.cli build-portfolio-html \
  --output reports/portfolio_public.html \
  --mode public \
  --check-privacy \
  --force

python scripts/check_public_portfolio_safety.py reports/portfolio_public.html
```

最後に、対象範囲、実行コマンド、成功件数、失敗件数、backup path、report path、
db-check結果、privacy check結果、次にやるべき手動reviewをdocsへ記録する。
