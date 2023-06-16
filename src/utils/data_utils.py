from scipy.io import loadmat
import os
import numpy as np
import pickle


def get_classes():
    return ['schooner','brain_coral','junco_bird','snail','grey_whale','siberian_husky','electric_fan','bookcase','fountain_pen','toaster','balance_beam',
                'school_bus','chainlink_fence','chime','coyote','aircraft_carrier','bubble','jellyfish','marmoset','wall_clock','water_snake','Welsh_springer_spaniel',
                'Arctic_fox','football_helmet','slug','potpie','Pomeranian','Indian_cobra','beach_wagon','Italian_greyhound','European_fire_salamander','chimpanzee',
                'typewriter_keyboard','black_and_gold_garden_spider','tick','toy_terrier','switch','lighter','guillotine','otterhound','boxer','hook','jersey',
                'soap_dispenser','umbrella','tiger_beetle','cash_machine','eel','Blenheim_spaniel','clumber']


def get_class_wids():
    return ['n04147183','n01917289','n01534433','n01944390','n02066245','n02110185','n03271574','n02870880','n03388183','n04442312','n02777292',  
                'n04146614','n03000134','n03017168','n02114855','n02687172','n09229709','n01910747','n02490219','n04548280','n01737021','n02102177','n02120079','n03379051',
                'n01945685','n07875152','n02112018','n01748264','n02814533','n02091032','n01629819','n02481823','n04505470','n01773157','n01776313','n02087046','n04372370',
                'n03666591','n03467068','n02091635','n02108089','n03532672','n03595614','n04254120','n04507155','n02165105','n02977058','n02526121','n02086646','n02101556']


def get_wid_labels(data_path,image_list):
    #Load the details of all the 1000 classes and the function to convert the synset id to words
    meta_clsloc_file = os.path.join(data_path,'meta_clsloc.mat')
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])

    #Code snippet to load the ground truth labels to measure the performance
    truth = {}
    with open(os.path.join(data_path,'ILSVRC2014_clsloc_validation_ground_truth.txt')) as f:
        line_num = 1
        for line in f.readlines():
            ind_ = int(line)
            temp  = None
            for i in synsets_imagenet_sorted:
                if i[0] == ind_:
                    temp = i
            #print ind_,temp
            if temp != None:
                truth[line_num] = temp
            else:
                print('##########', ind_)
                pass
            line_num += 1

    # Make list of wids
    true_valid_wids = []
    for i in image_list:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0].split('_')[-1]
        true_valid_wids.append(truth[int(temp)][1])
    true_valid_wids = np.asarray(true_valid_wids)

    return true_valid_wids


def get_labels(data_path,image_list):
    #Load the details of all the 1000 classes and the function to convert the synset id to words
    meta_clsloc_file = os.path.join(data_path,'meta_clsloc.mat')
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])

    #Code snippet to load the ground truth labels to measure the performance
    truth = {}
    with open(os.path.join(data_path,'ILSVRC2014_clsloc_validation_ground_truth.txt')) as f:
        line_num = 1
        for line in f.readlines():
            ind_ = int(line)
            temp  = None
            for idx,i in enumerate(synsets_imagenet_sorted):
                if i[0] == ind_:
                    temp = idx
            #print ind_,temp
            if temp != None:
                truth[line_num] = temp
            else:
                print('##########', ind_)
                pass
            line_num += 1

    # Make list of wids
    true_valid_wids = []
    for i in image_list:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0].split('_')[-1]
        true_valid_wids.append(truth[int(temp)])
    true_valid_wids = np.asarray(true_valid_wids)

    return true_valid_wids


def top5accuracy(true, predicted):
    """
    Function to predict the top 5 accuracy
    """
    assert len(true) == len(predicted)
    result = []
    flag  = 0
    for i in range(len(true)):
        flag  = 0
        temp = true[i]
        for j in predicted[i][0:5]:
            if j == temp:
                flag = 1
                break
        if flag == 1:
            result.append(1)
        else:
            result.append(0)
    counter = 0.
    for i in result:
        if i == 1:
            counter += 1.
    error = 1.0 - counter/float(len(result))
    #print len(np.where(np.asarray(result) == 1)[0])
    return len(np.where(np.asarray(result) == 1)[0]), error


class ImageWids:
    def __init__(self, path):
        self.__dict__ = pickle.load(open(path,'rb'))
    def __getitem__(self, key):
        if key in self.__dict__.keys():
            return self.__dict__[key]
        else:
            key = key.split('/')[-1].split('.')[0].split('_')[-1]
            return self.__dict__[key]