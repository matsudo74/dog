from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import numpy as np

input_filename = input("画像のパスを入力してください")

# 画像のサイズを224*224にしてリサイズしている
# load_imgには他にも関数あり
input_image = image.load_img(input_filename, target_size=(224, 224))

# カラー画像になっているので224*224*3の行列になっている
input_image = image.img_to_array(input_image)
# 四次元に拡張している
input_image = np.expand_dims(input_image, axis=0)
# 精度が向上する
input_image = preprocess_input(input_image)

# 既存モデルの導入
# 学習済みのVGG16
model = VGG16(weights="imagenet")

# 画像を予測
# predictは関数
results = model.predict(input_image)

# 予測結果とクラス名を紐付け（上位５クラスまで）
decode_results = decode_predictions(results, top=5)[0]

pred_ans = decode_results[0][1]
pred_score = decode_results[0][2]

#K.clear_session()
#一番予測結果の高い犬種と確率をreturn
return pred_ans, pred_score