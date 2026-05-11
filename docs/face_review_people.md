# Face Review and Manual People Labels

PR68 adds manual person labels for reviewed face clusters. This is a private,
human-in-the-loop review workflow. It is not automatic face recognition.

## Principles

- A person label is created only when the user types a name.
- The app never infers a name from a face.
- The app never links a face cluster to a LINE speaker automatically.
- Relationship labels such as partner, family, or friend are never inferred.
- Unreviewed clusters and unreviewed labels are not normal QA, search, event,
  report, or portfolio evidence.

## Data Model

`persons` stores manual labels:

- `display_name`: private local name
- `public_name`: optional anonymized name such as `人物A`
- `aliases_json`: local manual aliases
- `privacy_level`: `private`, `public_alias`, or `public_hidden`
- `manual_verified`: set for user-entered labels

`person_face_clusters` links a manual person label to a reviewed face cluster.
Links are manual and require explicit confirmation.

`person_aliases` stores manually added nicknames or aliases. PR69 can add a
LINE speaker name as an alias with source `line_speaker`, but only when the user
explicitly links that LINE speaker to the person.

`line_speaker_links` stores manual links from `(chat_id, speaker_name)` to a
person. The app never creates these links automatically from face clusters,
message text, or speaker names.

## CLI

```bash
python -m personal_lifelog_rag.app.cli persons list

python -m personal_lifelog_rag.app.cli persons create \
  --name "人物テストA" \
  --public-name "人物A" \
  --privacy-level private

python -m personal_lifelog_rag.app.cli persons show --person-id PERSON_ID

python -m personal_lifelog_rag.app.cli persons update \
  --person-id PERSON_ID \
  --name "表示名" \
  --public-name "人物A" \
  --privacy-level public_alias

python -m personal_lifelog_rag.app.cli persons add-alias \
  --person-id PERSON_ID \
  --alias "ニックネーム"

python -m personal_lifelog_rag.app.cli persons link-face-cluster \
  --person-id PERSON_ID \
  --cluster-id CLUSTER_ID \
  --yes

python -m personal_lifelog_rag.app.cli persons unlink-face-cluster \
  --person-id PERSON_ID \
  --cluster-id CLUSTER_ID \
  --yes

python -m personal_lifelog_rag.app.cli persons anonymize-preview
```

Use dummy names when testing the workflow. Real names should stay in the local
private DB only.

## UI Workflow

The local Gradio UI Face Review tab includes a cluster/person section:

1. Load face clusters.
2. Inspect the face-cluster thumbnail grid and select one cluster.
3. Enter a manual display name in "このクラスターの人物名".
4. Click "この名前でクラスターにラベル付け".
5. If the same display name already exists, the existing person is reused and
   the cluster is linked to that person. This is the only "same person" rule.
6. If the name is new, a new `persons` row is created and linked to the
   selected cluster.
7. Accept, reject, or mark a cluster as bad when needed.
8. Add aliases or update the privacy level.

The UI displays a reminder that names must be manual and that the app does not
infer identity or relationships.

PR69 also adds a LINE speaker link section to the Face Review tab:

1. Load local LINE speaker names.
2. Select a manual person label.
3. Enter `chat_id` and `speaker_name`.
4. Link or unlink the speaker manually.
5. Optionally add the speaker name as an alias.

Recommended simple flow:

1. Face clusters 読み込み.
2. サムネイルで同じ人らしいclusterを確認.
3. 同じ人物なら同じ表示名を入力.
4. LINE speakers 読み込み.
5. 該当するLINE話者を、選択中のpersonへ手動link.

## Public and Private Display

Private mode can show `display_name` for local review.

Public mode must not show private names:

- `private`: hidden by default
- `public_alias`: use `public_name`, or a generic label if missing
- `public_hidden`: hidden

Public reports and portfolio HTML must not include face crops, face embeddings,
cluster IDs, or private display names.

## Future Search/QA Use

PR69 enables only a narrow LINE speaker QA path such as
`人物テストAとLINEした日は？`. It uses manual `line_speaker_links` and returns
date-level message counts. Person photo search and richer person summaries
remain future opt-in work.

PR70 adds private `media_people` and `event_people` integration. These rows are
created only from manually verified person-face-cluster links and manually
linked LINE speakers. Unreviewed clusters, rejected clusters, bad detections,
and unverified persons are excluded.
