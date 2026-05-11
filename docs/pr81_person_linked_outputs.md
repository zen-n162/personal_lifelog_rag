# PR81 Person-Linked Outputs

This note records the PR81 integration of manually linked people into Ask,
image search, multimodal search, summaries, UI services, and private eval. It
uses anonymized examples only.

## Implemented

- Person resolver now checks display name, public name, aliases, JSON aliases,
  and manually linked LINE speaker names.
- Ask supports person-linked LINE, call, photo, activity, and place questions.
- Image Search and Multimodal Search include `person_score` and related person
  evidence when a person name is present in the query.
- Monthly Summary private mode includes compact manually linked person counts.
- Date/event answers can show related person candidates in private mode.
- UI services expose related person and person-evidence columns.
- Private eval can evaluate or skip person-linked QA safely.

## Rebuild Results

- backup: `backups/lifelog_before_pr81_rebuild_people_links_20260511_220452.sqlite`
- `build-media-people --replace`: inserted 434 rows from manual face-cluster links
- `build-event-people --replace`: inserted 3170 rows from manual LINE speaker links
- `people-stats`: persons total 5, linked face clusters 94, linked LINE speakers 2,
  media_people 434, event_people 3170

## Smoke Results

Anonymized smoke cases:

- `人物AとLINEした日は？`: pending
- `人物Aが写っている写真はいつ？`: pending
- `人物Aとご飯を食べた日は？`: pending
- `人物Aと新宿に行ったのはいつ？`: pending
- `2025年1月は何していた？`: pending
- `image-search "人物Aが写っている写真"`: pending
- `multimodal-search "人物Aとご飯"`: pending
- Ask LINE/call/photo/activity/place smoke: passed with manually linked person
  evidence and no relationship inference wording
- Monthly Summary smoke: private mode includes compact person-linked event,
  media, LINE, and call counts
- Image Search smoke: top results contain `related_persons` and `person_score`
- Multimodal Search smoke: top results contain `person_score`,
  `person_line_score`, and `person_event_score`
- Private eval template: 4 passed, 0 failed, 5 skipped because anonymized
  template entities were not registered locally

## Privacy Results

- `privacy-audit --public`: PASS
- `portfolio_public.html` safety check: PASS
- public person display names: not expected
- face crop / embedding exposure: not expected
- exact GPS exposure: not expected

## Remaining Work

- Review ambiguous duplicate person names manually in the Person/Face Review UI.
- Rebuild `media_people` and `event_people` after any person/cluster/speaker
  link changes.
- Keep public docs and portfolio output anonymized.
