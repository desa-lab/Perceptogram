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
# test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_cliptext1b_sub-BIGMEG1.npy', mmap_mode='r')
test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_cliptext1bcategory_sub-BIGMEG1.npy', mmap_mode='r')

subject = 'BIGMEG1'
# save_dir = 'cache/cliptext1b_module_weights/' + subject + '/'
save_dir = 'cache/cliptext1bcategory_module_weights/' + subject + '/'


all_pred_labels = np.zeros(test_labels.shape)
for i_token in range(77):
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

# np.save('cache/predicted_embeddings/BIGMEG1/thingsmeg_brainmodule_cliptext1b_sub-BIGMEG1.npy', all_pred_labels)
np.save('cache/predicted_embeddings/BIGMEG1/thingsmeg_brainmodule_cliptext1bcategory_sub-BIGMEG1.npy', all_pred_labels)

# Token 0: corr: 0.9999635798876587 euclidian dist: 0.0651144420465324                                                                                                      
# Token 1: corr: 0.929862408321571 euclidian dist: 0.3254427323141006                                                                                                       
# Token 2: corr: 0.6874052854300942 euclidian dist: 0.7600223822241539                                                                                                      
# Token 3: corr: 0.6058351702519826 euclidian dist: 0.773867813141459                                                                                                       
# Token 4: corr: 0.5807212679322841 euclidian dist: 0.7872058207713453                                                                                                      
# Token 5: corr: 0.5530418989228554 euclidian dist: 0.7883430369767936                                                                                                      
# Token 6: corr: 0.5246824052099897 euclidian dist: 0.7922491294905462                                                                                                      
# Token 7: corr: 0.5140553761046323 euclidian dist: 0.7889130681320973                                                                                                      
# Token 8: corr: 0.4906040677166542 euclidian dist: 0.792924323461032                                                                                                       
# Token 9: corr: 0.49106550502084323 euclidian dist: 0.8018977012536829                                                                                                     
# Token 10: corr: 0.5024632837828202 euclidian dist: 0.8154258407545405                                                                                                     
# Token 11: corr: 0.5403999044738792 euclidian dist: 0.8182578527771411                                                                                                     
# Token 12: corr: 0.5713192436803279 euclidian dist: 0.8151928488357469                                                                                                     
# Token 13: corr: 0.5954809401001184 euclidian dist: 0.8100799079706381                                                                                                     
# Token 14: corr: 0.6097738812105986 euclidian dist: 0.8064827368684839                                                                                                     
# Token 15: corr: 0.620422794508093 euclidian dist: 0.8032101082925078                                                                                                      
# Token 16: corr: 0.6272240709351727 euclidian dist: 0.8017305441668334                                                                                                     
# Token 17: corr: 0.631437392312897 euclidian dist: 0.8008528770320816                                                                                                      
# Token 18: corr: 0.6365718583625046 euclidian dist: 0.7990673029099367                                                                                                     
# Token 19: corr: 0.6416627014695953 euclidian dist: 0.7979069921457645                                                                                                     
# Token 20: corr: 0.6448532432065567 euclidian dist: 0.7970363159410208                                                                                                     
# Token 21: corr: 0.6505622574697961 euclidian dist: 0.795695190847371                                                                                                      
# Token 22: corr: 0.6545707144094287 euclidian dist: 0.7944499597844076                                                                                                     
# Token 23: corr: 0.6587351681273891 euclidian dist: 0.7934809053335128                                                                                                     
# Token 24: corr: 0.6604415775301801 euclidian dist: 0.7931332613871405                                                                                                     
# Token 25: corr: 0.6638216404413869 euclidian dist: 0.7915965630285575                                                                                                     
# Token 26: corr: 0.6632723829163599 euclidian dist: 0.7919173756000356                                                                                                     
# Token 27: corr: 0.6664396605152462 euclidian dist: 0.790758341083732                                                                                                      
# Token 28: corr: 0.6665265106239872 euclidian dist: 0.7908224320787521                                                                                                     
# Token 29: corr: 0.6674149247003184 euclidian dist: 0.7909176900418893                                                                                                     
# Token 30: corr: 0.6690628143948627 euclidian dist: 0.790279943534604
# Token 31: corr: 0.6692705834561129 euclidian dist: 0.7904246396188035
# Token 32: corr: 0.6698154853497005 euclidian dist: 0.7906163933556563
# Token 33: corr: 0.6687020601424172 euclidian dist: 0.7926839986566823
# Token 34: corr: 0.6720127605352962 euclidian dist: 0.789778201358633
# Token 35: corr: 0.6729168990326538 euclidian dist: 0.7899311644265918
# Token 36: corr: 0.6734751135256875 euclidian dist: 0.78920833230342
# Token 37: corr: 0.6743781771425004 euclidian dist: 0.7888861725677941
# Token 38: corr: 0.675426519352999 euclidian dist: 0.7886068976446309
# Token 39: corr: 0.6754451345925361 euclidian dist: 0.7884388880666539
# Token 40: corr: 0.676302497059174 euclidian dist: 0.7882864005623617
# Token 41: corr: 0.6776297029741994 euclidian dist: 0.7878727862578847
# Token 42: corr: 0.6786651385659677 euclidian dist: 0.7876731923649699
# Token 43: corr: 0.6803705547920536 euclidian dist: 0.7868400685517822
# Token 44: corr: 0.6816975406108297 euclidian dist: 0.7863643027123752
# Token 45: corr: 0.6843472943567942 euclidian dist: 0.7853146268173475
# Token 46: corr: 0.6861279635683983 euclidian dist: 0.7846029490380939
# Token 47: corr: 0.6883901803042343 euclidian dist: 0.7838184751533489
# Token 48: corr: 0.6887921013323122 euclidian dist: 0.7834299154542659
# Token 49: corr: 0.6899610563058374 euclidian dist: 0.7830709284186445
# Token 50: corr: 0.6908304241564516 euclidian dist: 0.7827915828727013
# Token 51: corr: 0.6928384389767566 euclidian dist: 0.7821050088389607
# Token 52: corr: 0.6936647759805711 euclidian dist: 0.7821553324157207
# Token 53: corr: 0.6955307623733 euclidian dist: 0.7809389090147696
# Token 54: corr: 0.6963769889930314 euclidian dist: 0.7806233057920989
# Token 55: corr: 0.6974127701106094 euclidian dist: 0.779705823301672
# Token 56: corr: 0.6995461904801569 euclidian dist: 0.7786316328625718
# Token 57: corr: 0.7005871871786786 euclidian dist: 0.7778099430787868
# Token 58: corr: 0.7021000418076778 euclidian dist: 0.7770336474522858
# Token 59: corr: 0.7035369623485169 euclidian dist: 0.7761863444263857
# Token 60: corr: 0.7048185609506422 euclidian dist: 0.7753894563438724
# Token 61: corr: 0.7065393460172706 euclidian dist: 0.7746097093557958
# Token 62: corr: 0.707015785108322 euclidian dist: 0.7744694839012272
# Token 63: corr: 0.7089720189946238 euclidian dist: 0.7731972013387916
# Token 64: corr: 0.7105601737657672 euclidian dist: 0.7723149286496793
# Token 65: corr: 0.7128563743812242 euclidian dist: 0.7707047444920806
# Token 66: corr: 0.713490698268724 euclidian dist: 0.7700624153657692
# Token 67: corr: 0.7131777301246408 euclidian dist: 0.7696692688719278
# Token 68: corr: 0.713563576701975 euclidian dist: 0.7693069656121113
# Token 69: corr: 0.7145525127095019 euclidian dist: 0.7685052779207123
# Token 70: corr: 0.7146262789762121 euclidian dist: 0.7681830101499071
# Token 71: corr: 0.7155666438859398 euclidian dist: 0.7678147978498285
# Token 72: corr: 0.7150466977015931 euclidian dist: 0.7684018658687556
# Token 73: corr: 0.7173317428922549 euclidian dist: 0.7663625678873587
# Token 74: corr: 0.7183595917320784 euclidian dist: 0.7654136272181095
# Token 75: corr: 0.7165021865825878 euclidian dist: 0.767029533710554
# Token 76: corr: 0.7218644153083648 euclidian dist: 0.7628313647146939

# Category
# Token 0: corr: 0.9999538711927183 euclidian dist: 0.05049881174002821                                                              
# Token 1: corr: 0.7045481851051405 euclidian dist: 0.5938705628806341                                                               
# Token 2: corr: 0.734028068271644 euclidian dist: 0.6236202274660023                                                                
# Token 3: corr: 0.8016649947096435 euclidian dist: 0.5848214745957114                                                               
# Token 4: corr: 0.8153666971922443 euclidian dist: 0.5736308889784784                                                               
# Token 5: corr: 0.8154739721854338 euclidian dist: 0.5714820848558411                                                               
# Token 6: corr: 0.8167973518226347 euclidian dist: 0.56647846265838                                                                 
# Token 7: corr: 0.8173701934663034 euclidian dist: 0.5622029685756996                                                               
# Token 8: corr: 0.8184450102674505 euclidian dist: 0.5572784234402136                                                               
# Token 9: corr: 0.8194613571746036 euclidian dist: 0.5522659673338416                                                               
# Token 10: corr: 0.8205304851436709 euclidian dist: 0.5481602895563876                                                              
# Token 11: corr: 0.8211616721414792 euclidian dist: 0.5444975570209535                                                              
# Token 12: corr: 0.8228363818389804 euclidian dist: 0.5400024627515465                                                              
# Token 13: corr: 0.8235042837983855 euclidian dist: 0.5373594683952714                                                              
# Token 14: corr: 0.8242715710348345 euclidian dist: 0.5344896407975781                                                              
# Token 15: corr: 0.8253612197089876 euclidian dist: 0.5318813138042654                                                              
# Token 16: corr: 0.8256888384254004 euclidian dist: 0.529945088725771                                                               
# Token 17: corr: 0.8257124574787185 euclidian dist: 0.5284396559615434                                                              
# Token 18: corr: 0.8257222386330215 euclidian dist: 0.5273526369827661                                                              
# Token 19: corr: 0.825839664277178 euclidian dist: 0.5260136399098321                                                               
# Token 20: corr: 0.8262038891976611 euclidian dist: 0.5251782916197891                                                              
# Token 21: corr: 0.8265778424179301 euclidian dist: 0.5241566349772387                                                              
# Token 22: corr: 0.8268916778258683 euclidian dist: 0.5232655453819013                                                              
# Token 23: corr: 0.8271206864094364 euclidian dist: 0.522514185284027                                                               
# Token 24: corr: 0.8271178968699346 euclidian dist: 0.5214963338333926                                                              
# Token 25: corr: 0.8272903964831682 euclidian dist: 0.5203999018113257                                                              
# Token 26: corr: 0.8269147041487347 euclidian dist: 0.5197611838023094                                                              
# Token 27: corr: 0.8264114243698759 euclidian dist: 0.5193166069703234                                                              
# Token 28: corr: 0.8252718943976313 euclidian dist: 0.5200431825507259                                                              
# Token 29: corr: 0.8256101196255876 euclidian dist: 0.518347916798945                                                               
# Token 30: corr: 0.8258243448418928 euclidian dist: 0.5174172983773482
# Token 31: corr: 0.825296412212536 euclidian dist: 0.5174238193483764
# Token 32: corr: 0.8253717030031883 euclidian dist: 0.5167915902249236
# Token 33: corr: 0.8253367925054266 euclidian dist: 0.5165717255422013
# Token 34: corr: 0.8251540482679199 euclidian dist: 0.515929169782673
# Token 35: corr: 0.8256434837319204 euclidian dist: 0.5146135221771846
# Token 36: corr: 0.8251862405648139 euclidian dist: 0.5141731881250132
# Token 37: corr: 0.8242617001232656 euclidian dist: 0.5141359530917484
# Token 38: corr: 0.8246264110344147 euclidian dist: 0.5130396535166142
# Token 39: corr: 0.8244840547081221 euclidian dist: 0.5123020305417992
# Token 40: corr: 0.8243963551573574 euclidian dist: 0.5119373376969221
# Token 41: corr: 0.8246837664346487 euclidian dist: 0.5115347785349057
# Token 42: corr: 0.8247022580507924 euclidian dist: 0.5109343588350282
# Token 43: corr: 0.825191031196044 euclidian dist: 0.5100636839221379
# Token 44: corr: 0.8257291521443973 euclidian dist: 0.5091376337223462
# Token 45: corr: 0.8261074868084347 euclidian dist: 0.5085956580785205
# Token 46: corr: 0.8265130393983645 euclidian dist: 0.5076451401946979
# Token 47: corr: 0.8270090566931568 euclidian dist: 0.5063405284382289
# Token 48: corr: 0.8266344027543572 euclidian dist: 0.5057081177833161
# Token 49: corr: 0.8260517126656786 euclidian dist: 0.5059594024305799
# Token 50: corr: 0.8266175403177491 euclidian dist: 0.5050412383889856
# Token 51: corr: 0.8273893211096972 euclidian dist: 0.503766170854671
# Token 52: corr: 0.8274167907858794 euclidian dist: 0.5037511162184551
# Token 53: corr: 0.8279959211570374 euclidian dist: 0.5025254143972474
# Token 54: corr: 0.828339568952311 euclidian dist: 0.5015595390346657
# Token 55: corr: 0.8289471403614399 euclidian dist: 0.500465350203493
# Token 56: corr: 0.8298184112656599 euclidian dist: 0.49899630801203365
# Token 57: corr: 0.830169575175463 euclidian dist: 0.4981762157572748
# Token 58: corr: 0.8300411532087171 euclidian dist: 0.49760052373561675
# Token 59: corr: 0.8310682844294079 euclidian dist: 0.4962942420639744
# Token 60: corr: 0.8317194620446344 euclidian dist: 0.49554413033147027
# Token 61: corr: 0.8322592541158982 euclidian dist: 0.4945090961019645
# Token 62: corr: 0.8330514943859567 euclidian dist: 0.49354023828651417
# Token 63: corr: 0.8338999231976018 euclidian dist: 0.4921798927905873
# Token 64: corr: 0.8340864380059921 euclidian dist: 0.49174229841170225
# Token 65: corr: 0.8350140906323549 euclidian dist: 0.4902610246162855
# Token 66: corr: 0.835200854957152 euclidian dist: 0.48920869924662497
# Token 67: corr: 0.8354185448934082 euclidian dist: 0.48819941745181133
# Token 68: corr: 0.8357025738476768 euclidian dist: 0.4869448295108261
# Token 69: corr: 0.8356187432420413 euclidian dist: 0.48685754626842426
# Token 70: corr: 0.8355556618822543 euclidian dist: 0.4866552303705521
# Token 71: corr: 0.83549602393016 euclidian dist: 0.4866137431520135
# Token 72: corr: 0.836649506090458 euclidian dist: 0.4848116113928116
# Token 73: corr: 0.8368530162964962 euclidian dist: 0.4843097055704758
# Token 74: corr: 0.8376656536367985 euclidian dist: 0.4833425976630503
# Token 75: corr: 0.8368342498521208 euclidian dist: 0.4825616202922248
# Token 76: corr: 0.840783648704649 euclidian dist: 0.4822086821030962