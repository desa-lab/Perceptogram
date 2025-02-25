from models import SimpleConv, BrainModule
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch import nn, optim
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import correlation
import os

test_data = np.load('cache/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1.npy', mmap_mode='r')
test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')

subject = 'BIGMEG1'
save_dir = 'cache/clipvision1b_module_weights/' + subject + '/'


all_pred_labels = np.zeros(test_labels.shape)
for i_token in range(257):
    device = torch.device('cuda:7')
    model = BrainModule()
    model.load_state_dict(torch.load(save_dir + f'{i_token}.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.to(device);

    test_data_tensor = torch.tensor(test_data).float()
    test_labels_tensor = torch.tensor(test_labels[:, i_token]).float()

    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Testing loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        preds = []
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
        pred_labels = np.array(preds)
        all_pred_labels[:, i_token] = pred_labels
    # Compute the Euclidean and correlation distances
    euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels[:, i_token])])
    correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels[:, i_token])])
    # Compute the average Euclidean distance
    average_euclidean_distance = euclidean_distances.mean()
    correlations = (1 - correlation_distances).mean()
    print(f'Token {i_token}:', 'corr:', correlations, 'euclidian dist:' ,average_euclidean_distance)

np.save('cache/predicted_embeddings/BIGMEG1/thingsmeg_brainmodule_clipvision1b_sub-BIGMEG1.npy', all_pred_labels)

# Token 0: corr: 0.741565380293411 euclidian dist: 0.6687740831234655                                                                     
# Token 1: corr: 0.6739744888098554 euclidian dist: 0.4660196553742867                                                                    
# Token 2: corr: 0.6698636446637077 euclidian dist: 0.4825302589725944                                                                    
# Token 3: corr: 0.6648802268793749 euclidian dist: 0.4794878566675597                                                                    
# Token 4: corr: 0.6675215327323071 euclidian dist: 0.479854789119277                                                                     
# Token 5: corr: 0.6678609174960826 euclidian dist: 0.4818135144692999                                                                    
# Token 6: corr: 0.6670860746690241 euclidian dist: 0.47996987763250537                                                                   
# Token 7: corr: 0.6600432842095317 euclidian dist: 0.4957469180607791                                                                    
# Token 8: corr: 0.6597110758657053 euclidian dist: 0.4862412987404043 
# Token 9: corr: 0.6623457787358482 euclidian dist: 0.4853336534100913 
# Token 10: corr: 0.6512653679581497 euclidian dist: 0.4896231314264527
# Token 11: corr: 0.6561126109297932 euclidian dist: 0.4911861910041902
# Token 12: corr: 0.6600315240987936 euclidian dist: 0.4911702077448147
# Token 13: corr: 0.6647111845580301 euclidian dist: 0.4863826137477906
# Token 14: corr: 0.6591102435567838 euclidian dist: 0.4811498982683552
# Token 15: corr: 0.6657326063345204 euclidian dist: 0.4867805741604055
# Token 16: corr: 0.6660658019189147 euclidian dist: 0.4768984643374908
# Token 17: corr: 0.6571743082078799 euclidian dist: 0.486927966041164 
# Token 18: corr: 0.6662369625806311 euclidian dist: 0.49037987270849537
# Token 19: corr: 0.6751650477691041 euclidian dist: 0.487350037993317 
# Token 20: corr: 0.6638028358581266 euclidian dist: 0.4973017103921575
# Token 21: corr: 0.6625956913706403 euclidian dist: 0.49724628637440066
# Token 22: corr: 0.6593445569650666 euclidian dist: 0.4987494186168383
# Token 23: corr: 0.6550433184797303 euclidian dist: 0.5025660688302244
# Token 24: corr: 0.6534415730243897 euclidian dist: 0.5010367893689069
# Token 25: corr: 0.6611573549158122 euclidian dist: 0.48740261965800374
# Token 26: corr: 0.6540567809981364 euclidian dist: 0.4996168501932205
# Token 27: corr: 0.6594942149277025 euclidian dist: 0.496791285270909 
# Token 28: corr: 0.6624249811645623 euclidian dist: 0.4935618869815799
# Token 29: corr: 0.6567524441970269 euclidian dist: 0.49485194549500744
# Token 30: corr: 0.6732614557656383 euclidian dist: 0.4809271744034174
# Token 31: corr: 0.667270035292479 euclidian dist: 0.48911313454402716
# Token 32: corr: 0.6539270284120365 euclidian dist: 0.4950797606471381
# Token 33: corr: 0.6612293330956267 euclidian dist: 0.45233914210146187
# Token 34: corr: 0.6556001798498033 euclidian dist: 0.49831803497508664
# Token 35: corr: 0.6692541656287746 euclidian dist: 0.4840854728033648
# Token 36: corr: 0.651688244104627 euclidian dist: 0.5017035509533355 
# Token 37: corr: 0.6517917262391312 euclidian dist: 0.5037310924345261
# Token 38: corr: 0.6544891133318774 euclidian dist: 0.5078517705968858
# Token 39: corr: 0.6455252355263985 euclidian dist: 0.5023422652407357
# Token 40: corr: 0.6425165113967908 euclidian dist: 0.5056082403844268
# Token 41: corr: 0.6450735454071412 euclidian dist: 0.5047478817333818
# Token 42: corr: 0.6359789683606787 euclidian dist: 0.5126402356753226
# Token 43: corr: 0.6533415176459644 euclidian dist: 0.4897634912911773
# Token 44: corr: 0.6446398014280567 euclidian dist: 0.5012497948677804
# Token 45: corr: 0.6620484400644697 euclidian dist: 0.4944056486106884
# Token 46: corr: 0.6631818402324732 euclidian dist: 0.4836378669455894
# Token 47: corr: 0.6579567951766734 euclidian dist: 0.49844617985581363
# Token 48: corr: 0.6404065580967758 euclidian dist: 0.4879083448945708
# Token 49: corr: 0.6571650064521796 euclidian dist: 0.48032171074516367
# Token 50: corr: 0.6596544562419063 euclidian dist: 0.4917615112156422
# Token 51: corr: 0.6620800693819843 euclidian dist: 0.4916714074192807
# Token 52: corr: 0.644333263753273 euclidian dist: 0.506346893051906
# Token 53: corr: 0.644532476223541 euclidian dist: 0.5082555717774735 
# Token 54: corr: 0.6415055742021613 euclidian dist: 0.5018428528364548
# Token 55: corr: 0.6300124995173805 euclidian dist: 0.5143722757152644
# Token 56: corr: 0.6382420291632322 euclidian dist: 0.5159390071503426
# Token 57: corr: 0.6372704682253696 euclidian dist: 0.5116757389979185
# Token 58: corr: 0.6375796882908736 euclidian dist: 0.5122036843639001
# Token 59: corr: 0.6396077146949508 euclidian dist: 0.5136606299401285
# Token 60: corr: 0.6370814154962837 euclidian dist: 0.5106186087180409
# Token 61: corr: 0.6499867867329273 euclidian dist: 0.49908604279848795
# Token 62: corr: 0.6608415287588058 euclidian dist: 0.48981889553299973
# Token 63: corr: 0.6532754827403666 euclidian dist: 0.49995324044217776
# Token 64: corr: 0.6476843347738246 euclidian dist: 0.49294606096013543
# Token 65: corr: 0.6517372078482866 euclidian dist: 0.46975632958278124
# Token 66: corr: 0.6535507975741327 euclidian dist: 0.4948414847182905
# Token 67: corr: 0.6420727651025885 euclidian dist: 0.5062127506962704
# Token 68: corr: 0.6431980209514259 euclidian dist: 0.5026251694979442
# Token 69: corr: 0.6369402576133278 euclidian dist: 0.5135366416499534
# Token 70: corr: 0.630679581530965 euclidian dist: 0.519603879417743
# Token 71: corr: 0.6330636834206745 euclidian dist: 0.5155373158416294
# Token 72: corr: 0.6285575463257604 euclidian dist: 0.5219186968700287
# Token 73: corr: 0.624955322234579 euclidian dist: 0.527132857802807
# Token 74: corr: 0.6275171290490477 euclidian dist: 0.5204832802756827
# Token 75: corr: 0.625332375504276 euclidian dist: 0.5207293468038364 
# Token 76: corr: 0.6353324813353184 euclidian dist: 0.5163509982326936
# Token 77: corr: 0.6450265434380927 euclidian dist: 0.5041596621056812
# Token 78: corr: 0.6567101181352897 euclidian dist: 0.49331971992142315
# Token 79: corr: 0.6486378028635784 euclidian dist: 0.5028540037444904
# Token 80: corr: 0.6392964701704937 euclidian dist: 0.5027237809931061
# Token 81: corr: 0.6506288509389518 euclidian dist: 0.47960092541727306
# Token 82: corr: 0.651205077357189 euclidian dist: 0.49595375855484275
# Token 83: corr: 0.6459169120227954 euclidian dist: 0.5040006782602795
# Token 84: corr: 0.6294980542687691 euclidian dist: 0.5144410291227146
# Token 85: corr: 0.6365083245381545 euclidian dist: 0.514044193747049 
# Token 86: corr: 0.6304136439359814 euclidian dist: 0.5230203294781527
# Token 87: corr: 0.617453315058284 euclidian dist: 0.5309611397015126 
# Token 88: corr: 0.6259100521450632 euclidian dist: 0.5259782431619525
# Token 89: corr: 0.6277525859265529 euclidian dist: 0.5264061530716316
# Token 90: corr: 0.6174044411421907 euclidian dist: 0.5318926855144835
# Token 91: corr: 0.6259336420518821 euclidian dist: 0.5218637395117309
# Token 92: corr: 0.6300151850424484 euclidian dist: 0.5141891329251236
# Token 93: corr: 0.6386929314754656 euclidian dist: 0.5110947268174203
# Token 94: corr: 0.6475627086084167 euclidian dist: 0.5028215501377474
# Token 95: corr: 0.6451270692688728 euclidian dist: 0.5020582906040513
# Token 96: corr: 0.6484043897093034 euclidian dist: 0.4874770312416204
# Token 97: corr: 0.6446817430710827 euclidian dist: 0.4871038141888738
# Token 98: corr: 0.6589021144537317 euclidian dist: 0.4982250002260961
# Token 99: corr: 0.6460614638817634 euclidian dist: 0.5102116282130763
# Token 100: corr: 0.6299379587129852 euclidian dist: 0.5178086786756501
# Token 101: corr: 0.6215867219107725 euclidian dist: 0.5247707322912509
# Token 102: corr: 0.6206953271783126 euclidian dist: 0.5251934540373159
# Token 103: corr: 0.620465537006008 euclidian dist: 0.5316641153663323
# Token 104: corr: 0.6285647806268717 euclidian dist: 0.5305719760162343
# Token 105: corr: 0.6194458162088959 euclidian dist: 0.534634180389611
# Token 106: corr: 0.6245029896892051 euclidian dist: 0.5333138428558052
# Token 107: corr: 0.6260399182053874 euclidian dist: 0.5242673655401622
# Token 108: corr: 0.6315412720477916 euclidian dist: 0.5239904794657954
# Token 109: corr: 0.6321004625071972 euclidian dist: 0.5155188099037542
# Token 110: corr: 0.6463596018532949 euclidian dist: 0.5003934572549829
# Token 111: corr: 0.6478331982014489 euclidian dist: 0.4989873744491782
# Token 112: corr: 0.6487865592558648 euclidian dist: 0.4970804204889235
# Token 113: corr: 0.6443891553133351 euclidian dist: 0.4878725915123401
# Token 114: corr: 0.6458406814904826 euclidian dist: 0.5077568219153004
# Token 115: corr: 0.645429462219664 euclidian dist: 0.5087386337601895
# Token 116: corr: 0.6388286786533359 euclidian dist: 0.5150657034991758
# Token 117: corr: 0.6287966567848059 euclidian dist: 0.5204725609047859
# Token 118: corr: 0.6236777049249366 euclidian dist: 0.5328568118791265
# Token 119: corr: 0.6180532441090856 euclidian dist: 0.5315538035242918
# Token 120: corr: 0.6303376263871964 euclidian dist: 0.5244317905608685
# Token 121: corr: 0.6230998294457034 euclidian dist: 0.5346759423351962
# Token 122: corr: 0.6218242147286952 euclidian dist: 0.5316170736069686
# Token 123: corr: 0.6143379490824921 euclidian dist: 0.5337073491763152
# Token 124: corr: 0.6208470096872466 euclidian dist: 0.5242716005638897
# Token 125: corr: 0.6221329965064636 euclidian dist: 0.5228933891328457
# Token 126: corr: 0.6395307828902848 euclidian dist: 0.5098635846188689
# Token 127: corr: 0.6481820365608938 euclidian dist: 0.5051578392433188
# Token 128: corr: 0.6414893369888517 euclidian dist: 0.4892662222306452
# Token 129: corr: 0.6537664873450875 euclidian dist: 0.4876004708675968
# Token 130: corr: 0.6482971619556148 euclidian dist: 0.5069630211077609
# Token 131: corr: 0.6377198644734441 euclidian dist: 0.5112440915743065
# Token 132: corr: 0.6285653486585612 euclidian dist: 0.5200724851806215
# Token 133: corr: 0.635857254233819 euclidian dist: 0.5182907198007503
# Token 134: corr: 0.6238635501794974 euclidian dist: 0.5332349753052998
# Token 135: corr: 0.6284419739611907 euclidian dist: 0.5277958849289295
# Token 136: corr: 0.6257677802628562 euclidian dist: 0.5252104261119208
# Token 137: corr: 0.6331388883097404 euclidian dist: 0.5234586230465984
# Token 138: corr: 0.6231453871403212 euclidian dist: 0.5267650399323973
# Token 139: corr: 0.6269151949196555 euclidian dist: 0.5279196526445983
# Token 140: corr: 0.6227100709535194 euclidian dist: 0.5234106954947695
# Token 141: corr: 0.6279933226902022 euclidian dist: 0.5223225924311665
# Token 142: corr: 0.6288092253785301 euclidian dist: 0.5184976347705168
# Token 143: corr: 0.6481676821756243 euclidian dist: 0.5043817354963702
# Token 144: corr: 0.6507257048822631 euclidian dist: 0.4943222808445148
# Token 145: corr: 0.6459856410688444 euclidian dist: 0.4902818320658103
# Token 146: corr: 0.6548125648520039 euclidian dist: 0.5017713152179701
# Token 147: corr: 0.6463874447636098 euclidian dist: 0.5110541960961213
# Token 148: corr: 0.635883144818153 euclidian dist: 0.5191449484437125
# Token 149: corr: 0.6294316197994897 euclidian dist: 0.5231373033376826
# Token 150: corr: 0.6248069761880691 euclidian dist: 0.5321342989407556
# Token 151: corr: 0.6254670266834139 euclidian dist: 0.5310355469999887
# Token 152: corr: 0.615388970699564 euclidian dist: 0.531626685550103 
# Token 153: corr: 0.6269945629335346 euclidian dist: 0.5250449824696515
# Token 154: corr: 0.6259373216203792 euclidian dist: 0.5300962670651012
# Token 155: corr: 0.6335667551182919 euclidian dist: 0.521515414799315
# Token 156: corr: 0.6221891857921779 euclidian dist: 0.5295168087024461
# Token 157: corr: 0.635245188153062 euclidian dist: 0.5184938513824928
# Token 158: corr: 0.6372664881504706 euclidian dist: 0.506112737501505
# Token 159: corr: 0.6549298840493583 euclidian dist: 0.5004351407583768
# Token 160: corr: 0.6427516790270801 euclidian dist: 0.49229979186994927
# Token 161: corr: 0.6442215477203911 euclidian dist: 0.48859076266702045
# Token 162: corr: 0.6579348161448538 euclidian dist: 0.49677368210748263
# Token 163: corr: 0.6493493653241199 euclidian dist: 0.5057884484885165
# Token 164: corr: 0.6372716406943028 euclidian dist: 0.5175768773653546
# Token 165: corr: 0.6424382596309912 euclidian dist: 0.5180185194102701
# Token 166: corr: 0.6217122921400795 euclidian dist: 0.5275569614863813
# Token 167: corr: 0.6225310187448658 euclidian dist: 0.531937986544953
# Token 168: corr: 0.6206097728317921 euclidian dist: 0.5235560653596597
# Token 169: corr: 0.6318931401226932 euclidian dist: 0.5257708676636229
# Token 170: corr: 0.6304201441325641 euclidian dist: 0.5265865666894187
# Token 171: corr: 0.6305933843507192 euclidian dist: 0.5231506524826517
# Token 172: corr: 0.6334463639565094 euclidian dist: 0.5208695952618162
# Token 173: corr: 0.6290549380223124 euclidian dist: 0.5214302364326883
# Token 174: corr: 0.6426964952188098 euclidian dist: 0.5123970337795796
# Token 175: corr: 0.6486708046511132 euclidian dist: 0.5089314359897544
# Token 176: corr: 0.6418660617576556 euclidian dist: 0.504016675028707
# Token 177: corr: 0.6466582872932024 euclidian dist: 0.48650333623290826
# Token 178: corr: 0.6387293893280207 euclidian dist: 0.5120494372714487
# Token 179: corr: 0.6577107814533246 euclidian dist: 0.4934350286995717
# Token 180: corr: 0.6547495966065126 euclidian dist: 0.5021628751169379
# Token 181: corr: 0.6507458230730757 euclidian dist: 0.5103117930394074
# Token 182: corr: 0.6390938435003043 euclidian dist: 0.5220912496551936
# Token 183: corr: 0.628697628540614 euclidian dist: 0.5217656503253553
# Token 184: corr: 0.6409675119421239 euclidian dist: 0.5227598898259498
# Token 185: corr: 0.6399260847241226 euclidian dist: 0.5169647997717952
# Token 186: corr: 0.6388937164909974 euclidian dist: 0.5157609807135392
# Token 187: corr: 0.6390317800119607 euclidian dist: 0.5179404251872588
# Token 188: corr: 0.6362740996213972 euclidian dist: 0.5195457149040832
# Token 189: corr: 0.6551272321010788 euclidian dist: 0.5005280269398065
# Token 190: corr: 0.6692274075059212 euclidian dist: 0.48966434325581626
# Token 191: corr: 0.6543666875120487 euclidian dist: 0.4964564195168339
# Token 192: corr: 0.6394279415976645 euclidian dist: 0.49857700972071956
# Token 193: corr: 0.6465317258347134 euclidian dist: 0.4867251463318763
# Token 194: corr: 0.6632881523413501 euclidian dist: 0.498374710461826
# Token 195: corr: 0.6569051946365618 euclidian dist: 0.4969566763981108
# Token 196: corr: 0.6552159204984194 euclidian dist: 0.5019920085173261
# Token 197: corr: 0.6553974111743274 euclidian dist: 0.5097749943051627
# Token 198: corr: 0.6544865846255646 euclidian dist: 0.506912608109587
# Token 199: corr: 0.6518355263993666 euclidian dist: 0.5120228961243289
# Token 200: corr: 0.6500064451758945 euclidian dist: 0.5092981503912533
# Token 201: corr: 0.6488218647230302 euclidian dist: 0.5048517588484066
# Token 202: corr: 0.6553648723908264 euclidian dist: 0.5092383709669526
# Token 203: corr: 0.6540099312811569 euclidian dist: 0.5023267453103136
# Token 204: corr: 0.6496903860421942 euclidian dist: 0.5096076167652732
# Token 205: corr: 0.6613235006403928 euclidian dist: 0.49555932934319363
# Token 206: corr: 0.6582988358410432 euclidian dist: 0.501591356666252
# Token 207: corr: 0.6601523041131824 euclidian dist: 0.487323313306487
# Token 208: corr: 0.660172894276932 euclidian dist: 0.48601511678882653
# Token 209: corr: 0.6445983915290113 euclidian dist: 0.47177526206145076
# Token 210: corr: 0.6668055217472181 euclidian dist: 0.493569941300122
# Token 211: corr: 0.6722539813525734 euclidian dist: 0.4935768992051965
# Token 212: corr: 0.6619137085371418 euclidian dist: 0.4982719548186892
# Token 213: corr: 0.6557673507095569 euclidian dist: 0.5011389850882212
# Token 214: corr: 0.6550909682207146 euclidian dist: 0.5052065403334965
# Token 215: corr: 0.6521010328204079 euclidian dist: 0.5066407256164185
# Token 216: corr: 0.6536718046193373 euclidian dist: 0.5025032400993177
# Token 217: corr: 0.6535386909757546 euclidian dist: 0.5027683193187799
# Token 218: corr: 0.6618941598008865 euclidian dist: 0.4978347843033748
# Token 219: corr: 0.6523317187838091 euclidian dist: 0.502554386319639
# Token 220: corr: 0.6586583135542681 euclidian dist: 0.501707566403601
# Token 221: corr: 0.6652098348731806 euclidian dist: 0.49456012470798344
# Token 222: corr: 0.6690612136291181 euclidian dist: 0.49351236033764545
# Token 223: corr: 0.6686769455321079 euclidian dist: 0.4960053027325449
# Token 224: corr: 0.6593945703842153 euclidian dist: 0.4829528373085668
# Token 225: corr: 0.6546506637203966 euclidian dist: 0.48431092938137266
# Token 226: corr: 0.672386748321299 euclidian dist: 0.492990912372583 
# Token 227: corr: 0.6795218696509524 euclidian dist: 0.4840965091845131
# Token 228: corr: 0.6752507128535171 euclidian dist: 0.49067318034845775
# Token 229: corr: 0.6622225018343711 euclidian dist: 0.4965200603709963
# Token 230: corr: 0.667571222606032 euclidian dist: 0.4856382211914022
# Token 231: corr: 0.6603635512063408 euclidian dist: 0.4958162524808349
# Token 232: corr: 0.6680268100790554 euclidian dist: 0.48921002243123135
# Token 233: corr: 0.658823210163495 euclidian dist: 0.49416338987292857
# Token 234: corr: 0.663958096514423 euclidian dist: 0.4918810553461958
# Token 235: corr: 0.6716443702861706 euclidian dist: 0.48553753472577 
# Token 236: corr: 0.6639181968175695 euclidian dist: 0.49287760621396237
# Token 237: corr: 0.663422879076309 euclidian dist: 0.48967042660297333
# Token 238: corr: 0.6785410513530497 euclidian dist: 0.47692012695008507
# Token 239: corr: 0.6752967778444222 euclidian dist: 0.4838914075911864
# Token 240: corr: 0.6649923928438831 euclidian dist: 0.48354688089948716
# Token 241: corr: 0.6783778393428354 euclidian dist: 0.45826969025192243
# Token 242: corr: 0.6830640239691058 euclidian dist: 0.4746020405642537
# Token 243: corr: 0.6789339559455468 euclidian dist: 0.4682149861785918
# Token 244: corr: 0.678188507639256 euclidian dist: 0.479757261738067 
# Token 245: corr: 0.6723258417644657 euclidian dist: 0.48128718349798194
# Token 246: corr: 0.6726612299884625 euclidian dist: 0.4877329104584355
# Token 247: corr: 0.6657060992114044 euclidian dist: 0.48563086746831613
# Token 248: corr: 0.6672821387880083 euclidian dist: 0.4870374457892897
# Token 249: corr: 0.6771579642190829 euclidian dist: 0.4783966442711729
# Token 250: corr: 0.6706356518503326 euclidian dist: 0.48167539267097825
# Token 251: corr: 0.6730454412377311 euclidian dist: 0.4733772884356
# Token 252: corr: 0.6782981166246147 euclidian dist: 0.47972225334398105
# Token 253: corr: 0.6731477997955486 euclidian dist: 0.4794304692815417
# Token 254: corr: 0.6856211410803172 euclidian dist: 0.4680544050426844
# Token 255: corr: 0.676630503019191 euclidian dist: 0.47877191492196874
# Token 256: corr: 0.6731467778903576 euclidian dist: 0.4633973853708515