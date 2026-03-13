# Tensory ロジックモデル整理

このドキュメントは、リポジトリ内コード全体を読み取った上で、ライブラリの設計モデルを「実装上の責務」と「実行時の流れ」に分けて整理したもの。

## 1. ワークスペース全体の役割分割

- tensory-core
	- 抽象の中心。TensorRepr / AxisMapper / Tensor / BoundTensor と、演算タスクの骨格を定義する。
- tensory-basic
	- 具体的な Id と Mapper 実装を供給する。デフォルト実装として VecMapper を持つ。
- tensory-linalg
	- SVD/QR/EIG/EXP/POW/NORM などの線形代数系タスクを、core 抽象の拡張として定義する。
- tensory-ndarray
	- ndarray + ndarray-linalg を使った実行バックエンド。core/linalg で定義した trait を具体実装する。
- tensory-regulated
	- 数値安定化のための係数分離モデル。テンソル本体と係数表現を分離して演算する。
- examples/*
	- API の使い方と、TRG/DMRG などのアルゴリズム実験。
- experiments/*
	- 性能測定、動的ロード、比較実験など。

要点は、core が純粋抽象、linalg が演算仕様、ndarray が実行実装、regulated が数値スケーリング戦略、というレイヤ分離になっていること。

## 2. コア概念: 3 層モデル

### Layer 1: TensorRepr

TensorRepr は、最も低い層のテンソル表現。

- 軸は 0..n-1 の usize で参照される。
- 同一オブジェクトに対して軸数と軸意味付けを変えてはならない。
- この不変条件のために unsafe trait として宣言されている。

つまり「データ表現そのもの」は TensorRepr で担い、意味付きの軸ラベルはまだ持たない。

### Layer 2: AxisMapper

AxisMapper は、usize 軸とユーザ ID の対応を持つ翻訳器。

- 1 つの mapper 内で ID は局所一意。
- TensorRepr の軸数と一致する必要がある。
- 対応関係は同一オブジェクト中で不変。

TensorRepr と AxisMapper を合成して、はじめて「ID で指定可能な軸」を持つ。

### Layer 3: Tensor / BoundTensor

Tensor は TensorRepr + AxisMapper の安全ラッパ。

- from_raw で repr.naxes == mapper.naxes を検証。
- view / view_mut で借用版テンソルを作る。
- ToTensor trait で所有/参照を統一的に扱う。

BoundTensor は Tensor + Runtime。

- runtime 一致チェック付きで演算を自動ディスパッチ。
- 「タスク生成は同じだが、どの Ctx で実行するか」を Runtime 側に寄せる。

## 3. 実行モデル: Task + Ctx + Runtime

Tensory の演算は基本的に次の 2 段階。

1. 演算子でタスクを作る
2. with(ctx) または exec() で実行する

例として加算は以下の流れ。

1. lhs + rhs
2. mapper を Overlay して軸対応を計算
3. TensorAdd タスクを生成
4. with(ctx) で AddCtxImpl::add_unchecked を呼ぶ
5. 結果 repr と結果 mapper から Tensor を再構築

実装上の厳密な型としては、加算演算子の戻り値は「TensorAdd 単体」ではなく、`Tensor<TensorAdd<...>, M>` になっている。
この形により、タスク本体の repr 部分と mapper を一体で保持したまま `with(ctx)` に渡せる。

この設計により、

- 演算仕様 (task 構築) と
- 実装戦略 (ctx 実装)

を分離できる。

BoundTensor では Runtime が add_ctx/mul_ctx/svd_ctx などを返し、ユーザは with を明示しなくても演算可能。

### 3.1 現行エラーモデル（運用上の注意）

現在の core 実装では、入力整合性チェックの一部が `panic!` ベースになっている。

- AddCtx / SubCtx / MulCtx / EwiseCtx の軸数不一致
- AxisInfo の範囲外アクセス

この設計は「内部契約違反は即時停止」という思想に沿っているが、アプリケーション入力由来の失敗を扱うには粗い。
今後は次の切り分けを採用すると良い。

- ライブラリ内部不変条件違反: debug_assert / panic
- ユーザ入力に起因する失敗: Result エラー

## 4. Mapper 代数 (軸情報の変換規約)

演算ごとに必要な軸変換情報を mapper trait 群で表現している。

- OverlayMapper
	- 加減算向け。2 テンソルの同型軸を重ねる。
- ConnectMapper
	- 縮約向け。共通 ID を接続軸ペアに変換する。
- GroupMapper / EquivGroupMapper
	- 分解向け。軸集合をグループに分割する。
- DecompGroupedMapper / SolveGroupedMapper
	- 分割後に新しいボンド ID を注入して各結果テンソルの mapper を再構築する。

SVD/QR/EIG が「どの軸を左群/右群に送るか」を mapper で安全に処理できるのは、この層の設計があるため。

## 5. tensory-basic の具体化

VecMapper が AxisMapper 群を実装する。

- 内部は Vec<Id>。
- 一意性チェックを保持。
- split/equiv_split/overlay/connect/replace/sort/translate を提供。

Id 側は Id128, Tag, Prime<Id> を提供。

- Prime は prime/deprime でレベル管理。
- アルゴリズム例では Prime を使って派生脚を表現する。

## 6. tensory-linalg の役割

linalg クレートは「分解演算のタスク仕様」を提供する。

- TensorSvd / TensorQr / TensorEig / TensorSolveEig / TensorExp / TensorPow / TensorNorm / TensorConj
- いずれも core の TensorTask パターンに従う。
- 分解系は GroupedAxes または EquivGroupedAxes を引数に持つ。

重要なのは、linalg 側は原則として「どの軸をどう扱うか」までを定義し、数値計算本体は backend context に委譲する点。

## 7. tensory-ndarray: 実行バックエンド

### 7.1 表現

- NdDenseRepr<E> = ArrayD<E>
- NdDenseViewRepr / NdDenseViewMutRepr = ndarray view

これらが TensorRepr / AsViewRepr / ElemGet / AxisInfo を実装する。

### 7.2 コンストラクタ

NdDenseTensorExt で以下を提供。

- zero / eye / random / random_hermite / random_unitary / scalar / map

mapper は BuildableMapper / SynBuildableMapper を使って同時に構築される。

### 7.3 演算 context 実装

arith.rs と linalg.rs で core/linalg trait を実装。

- add/sub: 軸 permutation 後に Zip で要素演算
- mul: 軸再配置して行列化、tenalg::mul で縮約
- svd/qr/eig/solve_eig: 行列化 -> ndarray-linalg -> 軸を戻して Tensor 化
- exp/pow (Diag): 対角成分に写像
- norm/conj も backend 実装を提供

NdRuntime は Runtime 実装で、各 Ctx を返すディスパッチャとして機能する。

### 7.4 Runtime とビルド要件

- `NdRuntime` は計算戦略の供給点であり、`add_ctx` / `mul_ctx` / `svd_ctx` などで実際の context を返す。
- テスト時には `__test-build` feature を要求するコンパイルガードがある。
	- これは BLAS/LAPACK のリンク要件を明示化するための仕組み。
	- CI とローカル手順の双方で有効化ポリシーを固定しておくと混乱が減る。

### 7.5 tenalg 補助層

tenalg モジュールは低レベル計算ユーティリティ。

- ten_to_mat / mat_to_ten
- mul
- svd / svddc / qr / eig / eigh
- diag_map
- singular value cut filter

core 抽象の外で数値処理をまとめ、context 実装を薄くしている。

## 8. tensory-regulated: 係数分離モデル

RegulatedRepr<A, C, N> は、

- repr: 正規化後テンソル本体
- coeff: 係数表現
- N: 規格化戦略

を持つ。

概念的には実テンソルを

T = coeff * repr

として扱う。

### 8.1 抽象

- Regulation / Regulator / Inflator
- CoefficientRepr
	- build, merge, mul, div, factorize
- ScalarRegulator

### 8.2 演算規則

- 乗算
	- 係数を merge し、結果 repr を再 regulate して係数に反映
- 加減算
	- coeff.factorize で共通スケールを抽出
	- weighted_add/sub で本体を合成
	- 再 regulate
- スカラー乗除
	- ScalarRegulator で phase/magnitude を分離して coeff 側へ集約

### 8.3 ndarray 側の具体例

- L2Regulator
	- L2 ノルムで repr を規格化
- LnCoeff
	- 係数を log で保持して桁落ちを抑制
	- factorize は最大 log を基準に取り、実係数比を exp で復元

この設計により、TRG のような反復で巨大/極小スケールが発生しても安定化しやすい。

## 9. 典型的な API 利用フロー

examples/usage と examples/algorithms から見える標準フローは次。

1. Tensor を leg-size map で生成
2. 演算子で task を構築
3. exec() か with(ctx) で実行
4. 分解系では leg split と新規 bond leg を指定
5. 必要なら bind(runtime) で BoundTensor 化
6. 長い反復では regulate して係数分離運用

## 10. 現状の実装ステータス

読解時点のコード上、以下は未実装または雛形段階。

- tensory-core arith/trace
- tensory-linalg lu, det
- examples/ideal は todo
- 一部 examples は骨組みのみ

加えて、`ElemGet` 系には「インデックス個数チェック」の TODO が残っている。

公開 API 観点では、未実装モジュールに対して次のどちらかを明示するとよい。

- 方針 A: 先行公開を維持し、ドキュメントに status=stub を明記
- 方針 B: feature gate または非公開化して、完成時に公開

ただし、コアのロジックモデル自体は一貫しており、以下が主軸。

- 抽象層の分離 (core)
- backend 実装注入 (ndarray)
- 数値安定化の直交拡張 (regulated)

## 11. 最短メンタルモデル

Tensory は「脚付きテンソル」を次の形で扱うフレームワーク。

- データ: TensorRepr
- 脚対応: AxisMapper
- 演算仕様: TensorTask
- 実行戦略: Ctx / Runtime
- 安定化: RegulatedRepr

つまり、

仕様を抽象化し、実装を差し替え可能にし、数値安定化を外付け可能にしたテンソルネット計算基盤。

## 12. 優先度付き改善計画

### P0（最優先）

- `ElemGet` / `ElemGetMut` の index 数チェックを実装し、境界契約を完成させる。
- 失敗モードを `panic` と `Result` に分類し、ユーザ入力由来の失敗を `Result` 側へ寄せる。

### P1（短期）

- 演算子実装の 9 パターン（owned/view/view_mut の直積）を共通化して保守コストを下げる。
- `trace` / `lu` / `det` の status を明確化（stub 明記か gate 化）。

### P2（中期）

- README とコードコメントの WIP/typo を解消し、ドキュメント品質を統一する。
- `develop.ai.md` を実装進捗に合わせて更新し、設計文書を living document 化する。
