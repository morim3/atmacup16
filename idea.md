# Atmacup 

## Problem


## Ideas
- Co-visitaion matrix (予測時に提案を生成）

- feature
-- embedding of hotel
--- セッションでの共起
--- 地域の埋め込み（session内での地域は近いはず)
-- セッション中に出てきたか？出てきたなら回数、順番
-- 値段帯を取り出したい
-- 時期
-- エリアごとに探索範囲の分散が異なる
--- ホテルがあまりない場所？, ネズミーランド
-- ~まで五分系の特徴量を補完

--- 人気志向かどうか、latest系特徴

--- matchのratio

-- Covisitation系特徴量(そのペアでなんかいcovisit, 何回next seq)
--他Sessionのarea cd?

## TODO
* latest_labelの特徴量過学習しそう



## candidate label ratio
- 20201211: 0.49
