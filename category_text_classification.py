
import pickle
import json 


def text_cate_predict(model_name, index_label_mapping, sample):

	# load the model from disk
	loaded_model = pickle.load(open(model_name, 'rb'))

	# load index label mapping
	with open(index_label_mapping) as json_file:
	    index_label = json.load(json_file)

	# predict with the loaded model
	result_index = loaded_model.predict([sample])[0]
	result = index_label[str(result_index)]

	return result


if __name__ == '__main__':

	model = 'text_model.pkl'
	index_label_mapping = 'index_label.json'
	samples = ['doctor consulting', 'room fees', 'glove']

	for sample in samples:
		predict = text_cate_predict(model, index_label_mapping, sample)
		print("The category prediction of ' "+sample+" ' is ' "+predict+" '")