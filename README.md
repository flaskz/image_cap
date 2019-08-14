# image_cap



## Dependencias
### Python 3.6.8
### Bibliotecas
- tensorflow==2.0.0-beta1
- matplotlib
- numpy
- pillow
- sklearn


### Parametros
- TEST_SIZE = 0.2
> Percentual de teste.

- BATCH_SIZE = 16
> Quantidade de imagens por batch.

- num_batches = 100
> Quantidade de batches de imagens para o treino.

- EPOCHS = 15
> Número de epochs do treino

- annotation_file
> Caminho para o arquivo de anotações do dataset coco

- PATH
> Caminho para o dataset coco

- checkpoint_path
> Caminho do diretório com o modelo treinado

##image_captioning.py
- train
> Define se vai treinar ou não (default=False)

- image_path
> Caminho da imagem para fazer predição (a pasta com o modelo treinado deve estar declarada no arquivo de parametros na variável "checkpoint_path")

- preprocess_images
> Define se as imagens do treino devem ser preprocessadas (o treino necessita que pelo menos uma vez as imagens sejam preprocessadas e salvas no disco, a partir do segundo treino com as mesmas imagens não é necessário pré-processamento)

- download_coco
> Baixa o dataset coco do link http://images.cocodataset.org/zips/train2014.zip e as anotações do link http://images.cocodataset.org/annotations/annotations_trainval2014.zip, descompacta e coloca no diretório do script (default=False)
