# メモ

## 設計

### `TensorRepr`

テンソルの最も低レベルな表現であることを保証するtrait。

axis(足)に対して、ローカルな`usize`で識別する。すなわち、$n$本の足に対して$0$から$n-1$までの`usize`を割り当てる。

同一の構造体はいかなる操作に対しても足の本数を変えず、また内部表現と足の対応を変えないことを実装で保証する必要がある。(例えば、サイズが同じだからといって多次元配列の足を入れ替えてはいけない。)これを`TensorRepr`の不変条件とする。

`TensorRepr`自体はあくまで表現であって、操作は`XxxCtx`で定義する。

### XxxCtx

操作Xxxを定義するtrait。

操作の実装自体はこのtraitを実装した構造体が持つ。この時構造体を消費することが許されるように設計するべきである。(つまり、`fn(self,...)`の形で入力を取るべき。)

足の識別に対する不変条件が入力に課される時は、`XxxCtxImpl`を設けて`unsafe`な関数も使えるようにしておく。

より高レベルの表現である`Tensor`・`BoundTensor`にラッパメソッドを実装するべきである。(外部crateの構造体に実装はできないので、ダミーのtraitを用意してその実装を記述するのが良い。)
ただし演算子による糖衣構文については`core`crateで実装されている関係上、`tensory-core`で一通りの定義をする。

### Tensor

足に対して追加の情報を用いて管理を行う`AxisMgr`を持つ。

#### AxisMgr

足に対する管理を行う。`AxisMgr`は管理対象の`TensorRepr`の足の本数を変えない。

##### LegMgr

足に対して`Id`を割り当て、操作に対して番号によらない指定方法を与える。

順序のない、**Id**で識別される**Leg**を持つ

### BoundTensor

`Tensor`にさらに演算に応じてXxxCtxを供給する`CtxProvider`への不変参照を張っている(可変性は内部可変を用いること)。

## rust

### unchecked(unsafe)に関する捉え方

- 通常の関数(すでに実装が確定している)
  - 関数の入力に型で表現されていない制約があるとき、
    - その関数はunsafeでマークされ、
    - 制約はdoc commentに記述されるべきで、
    - std crateでは Safetyなる項で記述される
    - e.g. `slice::get_unchecked`
    - 使用者は入力の制約を保証しなくてはならない
    - 実装者は入力の制約を前提として用いてよい
  - 関数の出力に型で表現されていない制約があるとき、
    - その関数はunsafeでマークされる必要はなく
    - 制約はdoc commentに記述されるべきだが、
    - std crateでは明示されないことも多い
    - e.g. `str::find` (出力のusizeは有効なindexを返すことが合意されているはずだが、明示はされていない)
    - 実装者は出力の制約を保証しなくてはならない
    - 使用者は出力の制約を前提として用いてよい
- traitの関数(実装を後から挿入できる)
  - 関数の入力に型で表現されていない制約があるとき、
    - その関数はunsafeでマークされ、
    - 制約はdoc commentに記述されるべきで、
    - std crateでは Safetyなる項で記述される
    - 使用者は入力の制約を保証しなくてはならない
    - 実装者は入力の制約を前提として用いて良い
  - 関数の出力に型で表現されていない制約があるとき、
    - その**トレイト**はunsafeでマークされ、
    - 制約はdoc commentに記述されるべきで、
    - std crateではおそらく書かれている
    - 実装者は出力の制約を保証しなくてはならない
    - 使用者は出力の制約を前提として用いてよい
    - `Sync`や`Send`はこの変種として見做せるだろう
    - [rust doc](https://doc.rust-lang.org/std/keyword.unsafe.html#unsafe-and-traits)を見よ

## grassmann

### なぜparityを固定するのか (特に、evenにするのか)

TRGは本来クソデカ積分の一部を積分するものだが、parityが混ざっているとテンソルを入れ替えるだけでsignがわさわさ出てくるので、よくない(特に、tensor graphはill-definedになる)

例えば \int ... T1 T2 T3 T4 ... = \int ... U1V1 U2V2 U3V3 U4V4 ... としても、実際に潰したいのはU1 U2 V3 V4だったりするので、交換が発生してワヤ

このことを考えるとevenかoddにしておくのが無難(evenの方が交換になってより良い)

### なぜSVDでnumericなbondを入れていないのか

parityが固定されないものがU,Vとして出てくるから、扱いが悪くなる

### 1, psi, psibar, psipsibarの構成は?

どうもauxなグラスマンを入れると等価っぽい

Koguts sequence https://ar5iv.labs.arxiv.org/html/2010.06539v2?utm_source=chatgpt.com#S2.SS1

Schwinger model

## CFT

CFT symbolic maybe use

NCTS 福住 PoChung Chen CFT TensorNet impled do email
