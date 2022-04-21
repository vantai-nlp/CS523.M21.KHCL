from deep_ir import *
from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/query', methods=['POST'])
def query():
    
    path = request.form.get('img_path')
    img = Utils.load_img(path)
    start_time = time.time()

    input_query = feature_extractor.extract(img.reshape(1, *img.shape))
    indexs, _ = matching_model.query(input_query)

    imgs = []
    for index in indexs:
        imgs.append(img_paths[index])
    print(f'[INFO] Time quering = {time.time()-start_time}') 

    reponse = {'result': imgs}
    return jsonify(reponse)


if __name__ == '__main__':

    # init models
    feature_extractor = Feature_Extractor_VGG16()

    # load database
    features, img_paths = Utils.load_database('./database_feature_extraction.pkl')

    # init matching model
    matching_model = Matching(features)  
    
    app.run(debug=True, host='127.0.0.2', port='2222')
