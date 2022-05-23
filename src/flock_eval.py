# Evaluate the 'flock' performance
# See report section 5.3

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Whole slide
# Confidences generated by alpha models
all_alpha_confidences = [
    # alpha no normal
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0665740966796875, 0.0, 0.0, 0.0, 0.0001550912857055664, 0.9999548196792603, 0.0, 1.1920928955078125e-07, 2.753734588623047e-05, 0.0, 0.9001526236534119, 0.0, 1.1920928955078125e-07, 0.9999982118606567, 0.9996938705444336, 0.0, 0.9890584349632263, 0.8146665692329407, 1.0, 0.00024431943893432617, 0.9976927042007446, 0.9999779462814331, 0.0, 3.5762786865234375e-07, 0.0, 0.0, 0.0, 0.0, 0.9999814033508301, 0.0, 0.8023349642753601, 1.0, 0.9827919602394104, 0.0, 0.0, 0.9999778270721436, 0.0, 0.0, 0.0, 0.0, 3.5762786865234375e-07, 0.0, 0.0, 0.9784099459648132,
        0.0, 0.0, 1.0, 0.0, 0.9890437126159668, 1.0, 0.16612839698791504, 1.0, 1.0, 1.0, 1.0, 0.0, 0.00024896860122680664, 1.0, 1.0, 0.00046241283416748047, 0.0030518174171447754, 1.919269561767578e-05, 0.005326449871063232, 0.0, 1.0, 1.0, 7.104873657226562e-05, 0.999066174030304, 7.3909759521484375e-06, 0.9999923706054688, 1.0, 0.9814815521240234, 0.0, 0.0003229379653930664, 0.9955776929855347, 1.0, 1.0, 1.0, 1.0, 0.0, 2.7060508728027344e-05, 1.0, 0.0, 2.0265579223632812e-06, 0.9999991655349731, 1.0, 0.9092322587966919, 0.0, 0.0, 0.9999970197677612, 1.0, 1.0, 1.0, 0.005230247974395752],
    # alpha normal
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.960464477539062e-07, 0.0, 1.0, 1.0, 0.9524440169334412, 0.9900907278060913, 0.00973498821258545, 0.0, 1.0, 0.0, 1.8596649169921875e-05, 1.0, 0.9981740713119507, 0.00034165382385253906, 0.9999971389770508, 1.0, 1.0, 0.9999953508377075, 0.9999972581863403, 1.0, 0.0, 4.744529724121094e-05, 0.0, 0.0, 0.7905524969100952, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 7.152557373046875e-07, 0.0, 0.0,
        0.002952754497528076, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.00035262107849121094, 1.0, 1.0, 1.0, 0.9999970197677612, 9.930133819580078e-05, 0.001695394515991211, 0.0, 1.0, 1.0, 0.8609916567802429, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.9999501705169678, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 5.9604644775390625e-06, 1.0, 0.0, 2.9325485229492188e-05, 1.0, 1.0, 0.08691287040710449, 1.1920928955078125e-07, 0.9999998807907104, 1.0, 1.0, 1.0, 1.0, 1.0],
    # alpha a no normal
    [0.06489771604537964, 0.01930856704711914, 0.009812414646148682, 0.018959879875183105, 0.06330424547195435, 0.0055122971534729, 0.3225237727165222, 0.05600625276565552, 0.0729759931564331, 0.043185293674468994, 0.05755317211151123, 0.35619300603866577, 0.053370773792266846, 0.05868464708328247, 0.010007679462432861, 0.060333967208862305, 0.029230475425720215, 0.05905693769454956, 0.05087703466415405, 0.1194499135017395, 0.3540421724319458, 0.0572170615196228, 0.05036890506744385, 0.006588160991668701, 0.6724592447280884, 0.05007857084274292, 0.04138106107711792, 0.014710485935211182, 0.05870389938354492, 0.0062133073806762695, 0.06566959619522095, 0.050166964530944824, 0.058707237243652344, 0.01945871114730835, 0.015468955039978027, 0.05984312295913696, 0.04659169912338257, 0.826529860496521, 0.7888273596763611, 0.008371710777282715, 0.059431374073028564, 0.890762984752655, 0.04721713066101074, 0.05039340257644653, 0.012882769107818604, 0.05600440502166748, 0.02431666851043701, 0.0032750964164733887, 0.058457255363464355,
        0.030020475387573242, 0.003895401954650879, 0.055860161781311035, 0.5598635673522949, 0.7610422968864441, 0.7622742056846619, 0.21379417181015015, 0.1744256615638733, 0.9656199812889099, 0.8827544450759888, 0.8266375064849854, 0.7494859099388123, 0.016873300075531006, 0.006124556064605713, 0.8301898837089539, 0.9617867469787598, 0.05298411846160889, 0.16126519441604614, 0.0067359209060668945, 0.7547471523284912, 0.0014996528625488281, 0.5052740573883057, 0.14574378728866577, 0.06560468673706055, 0.9683659076690674, 0.9747171998023987, 0.982296884059906, 0.26492100954055786, 0.01528024673461914, 0.01085585355758667, 0.06349945068359375, 0.04068613052368164, 0.23590004444122314, 0.7640999555587769, 0.6241065859794617, 0.3515661954879761, 0.05066120624542236, 0.009745419025421143, 0.40104877948760986, 0.04606050252914429, 0.02647930383682251, 0.011384189128875732, 0.9088724255561829, 0.7950494289398193, 0.029656946659088135, 0.9703430533409119, 0.5879155397415161, 0.8218039274215698, 0.8218039274215698, 0.37776827812194824, 0.6222317218780518],
    # alpha a normal
    [0.09672224521636963, 0.004344344139099121, 0.020706474781036377, 0.014834880828857422, 0.06451523303985596, 0.0023360252380371094, 0.970547616481781, 0.014928460121154785, 0.18154174089431763, 0.009385824203491211, 0.12811028957366943, 0.41215407848358154, 0.007848739624023438, 0.017266511917114258, 0.018698811531066895, 0.04201161861419678, 0.008692800998687744, 0.07059186697006226, 0.22944164276123047, 0.49692004919052124, 0.15257686376571655, 0.06750786304473877, 0.007293045520782471, 0.006710827350616455, 0.5660027265548706, 0.8402100205421448, 0.0037251710891723633, 0.059774935245513916, 0.027859628200531006, 0.010539114475250244, 0.03449040651321411, 0.039497435092926025, 0.018019020557403564, 0.021445095539093018, 0.016276836395263672, 0.03655022382736206, 0.3307902216911316, 0.7591058015823364, 0.6310852766036987, 0.0005347132682800293, 0.0784652829170227, 0.9173064827919006, 0.0069977641105651855, 0.177665114402771, 0.0033947229385375977, 0.024437546730041504, 0.0031875967979431152, 0.0023146867752075195, 0.027630984783172607,
        0.05476200580596924, 0.0014630556106567383, 0.44247496128082275, 0.2774720788002014, 0.7442097067832947, 0.4942557215690613, 0.19502317905426025, 0.8799218535423279, 0.9504480361938477, 0.8644542694091797, 0.4056550860404968, 0.4699627757072449, 0.01257997751235962, 0.009833753108978271, 0.5595703721046448, 0.9421853423118591, 0.012765586376190186, 0.9150501489639282, 0.0028235912322998047, 0.7392506003379822, 0.0008453130722045898, 0.5115358233451843, 0.18531358242034912, 0.15138864517211914, 0.9457085728645325, 0.9812989830970764, 0.9872045516967773, 0.24195915460586548, 0.10943335294723511, 0.0036172866821289062, 0.045295000076293945, 0.0009272098541259766, 0.19149619340896606, 0.8085038065910339, 0.6228235960006714, 0.28773999214172363, 0.01603519916534424, 0.01600426435470581, 0.41112738847732544, 0.02818530797958374, 0.037925124168395996, 0.002834022045135498, 0.9410603046417236, 0.9206077456474304, 0.03810304403305054, 0.9618969559669495, 0.6470398902893066, 0.25094759464263916, 0.7490524053573608, 0.2624405026435852, 0.7375594973564148],
    # alpha n no normal
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1920928955078125e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2.2649765014648438e-06, 0.0, 0.0, 0.0, 0.0, 5.960464477539062e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.76837158203125e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.1920928955078125e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.1920928955078125e-07, 0.9999998807907104],
    # alpha n normal
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    # alpha a n no normal
    [0.15288621187210083, 9.97781753540039e-05, 0.001008450984954834, 0.011824727058410645, 0.10420036315917969, 3.790855407714844e-05, 0.005465984344482422, 0.03930610418319702, 5.7816505432128906e-05, 0.00018328428268432617, 0.010996997356414795, 0.9999270439147949, 0.0081099271774292, 0.14712554216384888, 0.00018721818923950195, 0.10912644863128662, 8.809566497802734e-05, 0.18841248750686646, 0.0008620023727416992, 0.016077876091003418, 0.007323026657104492, 0.014733076095581055, 0.9704116582870483, 0.009548425674438477, 0.9988937973976135, 0.013564646244049072, 0.0002982020378112793, 0.00018388032913208008, 0.13986480236053467, 0.00022560358047485352, 0.0034766793251037598, 0.00045883655548095703, 0.007902979850769043, 1.1324882507324219e-05, 0.706486165523529, 0.0001887679100036621, 0.00030219554901123047, 0.893608808517456, 0.9846146702766418, 1.9073486328125e-06, 0.18012768030166626, 0.04930675029754639, 0.07083702087402344, 0.0006681084632873535, 4.7087669372558594e-05, 0.19092172384262085, 5.0902366638183594e-05, 4.649162292480469e-06,
        0.18382608890533447, 0.000491023063659668, 2.3484230041503906e-05, 4.8160552978515625e-05, 0.009243786334991455, 0.6626598238945007, 0.9999874830245972, 0.9998797178268433, 0.00027745962142944336, 0.9973402619361877, 0.9983437061309814, 0.02253425121307373, 0.9871642589569092, 0.00021511316299438477, 0.00014066696166992188, 0.9999889135360718, 1.0, 0.000867307186126709, 0.00099867582321167, 1.609325408935547e-05, 0.9999551773071289, 0.37723445892333984, 1.0, 0.9999756813049316, 6.92605972290039e-05, 0.20121657848358154, 0.0016651749610900879, 0.9799320101737976, 0.9999861717224121, 0.0003045797348022461, 2.384185791015625e-06, 0.00011050701141357422, 0.0387500524520874, 0.9999997615814209, 0.9999997615814209, 0.9999881982803345, 1.0, 0.002657949924468994, 3.4570693969726562e-06, 0.3218556046485901, 0.04751908779144287, 0.011127889156341553, 0.1802922487258911, 0.4902195930480957, 0.0016494989395141602, 0.877873420715332, 0.877873420715332, 0.013442456722259521, 0.9982473850250244, 0.9982473850250244, 0.9995085000991821, 0.9995085000991821],
    # alpha a n normal
    [0.000539243221282959, 0.027276277542114258, 0.9415848851203918, 0.8172564506530762, 0.0005927085876464844, 5.364418029785156e-06, 0.9999998807907104, 0.0018128752708435059, 0.00239640474319458, 0.00022774934768676758, 0.9999997615814209, 1.0, 0.0006273388862609863, 0.24839234352111816, 0.009084045886993408, 0.000594019889831543, 0.9647147059440613, 0.004099905490875244, 0.0012985467910766602, 0.9999455213546753, 0.44794797897338867, 8.916854858398438e-05, 0.9999998807907104, 0.9794146418571472, 0.9999998807907104, 0.9993256330490112, 0.008789420127868652, 0.009988367557525635, 0.02013528347015381, 0.4503554701805115, 0.9629515409469604, 0.3773176074028015, 0.3897383213043213, 0.0008521080017089844, 0.9997276663780212, 0.05727994441986084, 1.0, 0.9993399977684021, 1.0, 3.5762786865234375e-07, 0.0066356658935546875, 0.9092878699302673, 0.0016672015190124512, 0.0023046135902404785, 7.49826431274414e-05, 0.0011275410652160645, 0.0001266002655029297,
        0.00017434358596801758, 0.0820799469947815, 0.565701961517334, 9.071826934814453e-05, 0.000871121883392334, 0.8133051991462708, 0.9988052845001221, 1.0, 0.9999980926513672, 0.9999994039535522, 0.9999983310699463, 1.0, 0.9999966621398926, 0.9999991655349731, 0.835020124912262, 0.005586445331573486, 0.9999992847442627, 1.0, 0.0014717578887939453, 0.6297593712806702, 0.00011801719665527344, 0.9999988079071045, 0.9775444269180298, 1.0, 0.9999971389770508, 0.04441887140274048, 0.9998407363891602, 0.034990549087524414, 0.9999077320098877, 0.9999982118606567, 0.9953670501708984, 0.0007222294807434082, 0.9805924892425537, 0.9999973773956299, 1.0, 1.0, 0.9999997615814209, 1.0, 0.0008493661880493164, 0.013252615928649902, 0.9842639565467834, 0.0002270340919494629, 0.600939154624939, 0.9512985944747925, 1.0, 0.5454967617988586, 0.9947053790092468, 0.9947053790092468, 1.0, 0.9999998807907104, 0.9999998807907104, 0.999998927116394, 0.999998927116394]
]

# Confidence threshold of 0.5, i.e., round to nearest whole number (0 or 1)
all_alpha_classifications = []
for confidences in all_alpha_confidences:
    rounded = [round(x) for x in confidences]
    all_alpha_classifications.append(rounded)

average_alpha_classification = [
    0 for i in range(len(all_alpha_confidences[0]))]
for i in range(len(all_alpha_classifications)):
    for j in range(len(all_alpha_classifications[0])):
        average_alpha_classification[j] += all_alpha_classifications[i][j]

for i in range(len(average_alpha_classification)):
    average_alpha_classification[i] = average_alpha_classification[i] / \
        len(all_alpha_classifications)

average_rounded = []
for confidence in average_alpha_classification:
    rounded = round(confidence)
    average_rounded.append(rounded)

true_alpha_classification = [
    0 for i in range(0, 50)] + [1 for i in range(0, 50)]

m = tf.keras.metrics.AUC(num_thresholds=100)
m.update_state(true_alpha_classification, average_alpha_classification)
print("AUC:", m.result().numpy())

tp = 0
fp = 0
tn = 0
fn = 0

print(average_rounded)
print(true_alpha_classification)

for i in range(len(average_rounded)):
    if average_rounded[i] == 1:
        if true_alpha_classification[i] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if true_alpha_classification[i] == 0:
            tn += 1
        else:
            fn += 1

print(tp, fp, tn, fn)

print("Accuracy:", (tp + tn) / (tp + fp + tn + fn))
print("Specificity:", tn / (tn + fp))
print("Sensitivity:", tp / (tp + fn))

# Adapated from https://androidkt.com/get-the-roc-curve-and-auc-for-keras-model/


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.axis('square')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# False positive rate (1-true negative rate) of alpha_a_n
fpr = []
tpr = []
step = 0.01
for i in np.arange(0.0, 1 + step, step):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(all_alpha_confidences[7])):
        if all_alpha_confidences[7][j] < i:
            if true_alpha_classification[j] == 0:
                tn += 1
            else:
                fn += 1
        else:
            if true_alpha_classification[j] == 1:
                tp += 1
            else:
                fp += 1
    tpr.append(tp / (tp + fn))
    fpr.append(1 - (tn / (tn + fp)))

fpr[100] = 0
tpr[100] = 0
plot_roc_curve(fpr, tpr)
