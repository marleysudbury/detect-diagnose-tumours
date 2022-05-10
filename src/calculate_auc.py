# Calculate AUC from classification classification_confidences

import tensorflow as tf

# classification_confidences = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0665740966796875, 0.0, 0.0, 0.0, 0.0001550912857055664, 0.9999548196792603, 0.0, 1.1920928955078125e-07, 2.753734588623047e-05, 0.0, 0.9001526236534119, 0.0, 1.1920928955078125e-07, 0.9999982118606567, 0.9996938705444336, 0.0, 0.9890584349632263, 0.8146665692329407, 1.0, 0.00024431943893432617, 0.9976927042007446, 0.9999779462814331, 0.0, 3.5762786865234375e-07, 0.0, 0.0, 0.0, 0.0, 0.9999814033508301, 0.0, 0.8023349642753601, 1.0, 0.9827919602394104, 0.0, 0.0, 0.9999778270721436, 0.0, 0.0, 0.0, 0.0, 3.5762786865234375e-07, 0.0, 0.0, 0.9784099459648132, 0.0, 0.0, 1.0, 0.0, 0.9890437126159668, 1.0, 0.16612839698791504, 1.0, 1.0, 1.0, 1.0, 0.0, 0.00024896860122680664, 1.0, 1.0, 0.00046241283416748047, 0.0030518174171447754, 1.919269561767578e-05, 0.005326449871063232, 0.0, 1.0, 1.0, 7.104873657226562e-05, 0.999066174030304, 7.3909759521484375e-06, 0.9999923706054688, 1.0, 0.9814815521240234, 0.0, 0.0003229379653930664, 0.9955776929855347, 1.0, 1.0, 1.0, 1.0, 0.0, 2.7060508728027344e-05, 1.0, 0.0, 2.0265579223632812e-06, 0.9999991655349731, 1.0, 0.9092322587966919, 0.0, 0.0, 0.9999970197677612, 1.0, 1.0, 1.0, 0.005230247974395752]

# classification_confidences = [0.06343233585357666, 0.024738788604736328, 0.014714419841766357, 0.021375000476837158, 0.06335914134979248, 0.0063934326171875, 0.11911451816558838, 0.05293428897857666, 0.08082520961761475, 0.046549975872039795, 0.05783504247665405, 0.25085288286209106, 0.0546303391456604, 0.05892902612686157, 0.012165427207946777, 0.06125926971435547, 0.032364845275878906, 0.059192776679992676, 0.05079871416091919, 0.09894371032714844, 0.3084195852279663, 0.05734360218048096, 0.06436610221862793, 0.005237162113189697, 0.7127169370651245, 0.05075746774673462, 0.043930768966674805, 0.006013989448547363, 0.05858325958251953, 0.007522106170654297, 0.0610920786857605, 0.04672062397003174, 0.06343328952789307, 0.01017141342163086, 0.01472175121307373, 0.058140575885772705, 0.04507941007614136, 0.79871666431427, 0.6649472117424011, 0.008872687816619873, 0.05984652042388916, 0.7938653826713562, 0.0529516339302063, 0.04305034875869751, 0.010911166667938232, 0.0562397837638855, 0.030173122882843018, 0.0029616355895996094, 0.058045268058776855, 0.031154990196228027, 0.0037357211112976074, 0.041092872619628906, 0.5719906687736511, 0.6343598961830139, 0.7691723704338074, 0.15876144170761108, 0.1175050139427185, 0.9737184643745422, 0.9042090773582458, 0.8610864877700806, 0.4528180956840515, 0.0208091139793396, 0.0055335164070129395, 0.8210203647613525, 0.9574425220489502, 0.05376309156417847, 0.10014301538467407, 0.005984961986541748, 0.7268585562705994, 0.0015425682067871094, 0.5381443500518799, 0.14928871393203735, 0.05162233114242554, 0.9653250575065613, 0.9675791263580322, 0.9831672310829163, 0.19343596696853638, 0.022676289081573486, 0.013596892356872559, 0.06884640455245972, 0.035190463066101074, 0.2624141573905945, 0.24036073684692383, 0.6443600654602051, 0.3130255341529846, 0.050923824310302734, 0.011345863342285156, 0.21100842952728271, 0.04673135280609131, 0.02876073122024536, 0.030574798583984375, 0.785302996635437, 0.9182453155517578, 0.029656946659088135, 0.013530969619750977, 0.5508366823196411, 0.8688876628875732, 0.03578078746795654, 0.4449141025543213, 0.12014555931091309]

# classification_confidences = [0.16860777139663696, 0.0013724565505981445, 0.0002447962760925293, 0.012830615043640137, 0.1424674391746521, 3.170967102050781e-05, 0.0008612871170043945, 0.054916560649871826, 7.450580596923828e-05, 0.00019365549087524414, 0.02155238389968872, 0.9717340469360352, 0.0043828487396240234, 0.15854549407958984, 0.0001277923583984375, 0.16056126356124878, 6.413459777832031e-05, 0.18842428922653198, 0.0038129687309265137, 0.009870469570159912, 0.002646207809448242, 0.049604058265686035, 0.8752835988998413, 0.0018184185028076172, 0.9949131011962891, 0.012134253978729248, 0.0002180933952331543, 0.0002461075782775879, 0.1341848373413086, 0.00012576580047607422, 0.02623283863067627, 0.0010344982147216797, 0.06528604030609131, 1.3947486877441406e-05, 0.3244466185569763, 7.319450378417969e-05, 0.00027638673782348633, 0.9977126121520996, 0.9401959776878357, 3.4570693969726562e-06, 0.1833474040031433, 0.007618367671966553, 0.12688416242599487, 0.0010477304458618164, 3.337860107421875e-05, 0.19165539741516113, 4.887580871582031e-05, 3.0994415283203125e-06, 0.18767035007476807, 0.0006392598152160645, 2.6226043701171875e-05, 7.128715515136719e-05, 0.0006978511810302734, 0.12940144538879395, 0.9976208806037903, 0.999990701675415, 0.0002664923667907715, 0.9685708284378052, 0.998379111289978, 0.0038163065910339355, 0.9800091981887817, 0.0001474618911743164, 0.0001354217529296875, 0.9998928308486938, 1.0, 0.004886806011199951, 0.0007405877113342285, 1.823902130126953e-05, 0.9998784065246582, 0.18460851907730103, 1.0, 0.9999247789382935, 7.033348083496094e-05, 0.4661285877227783, 0.0018961429595947266, 0.991304874420166, 0.9999656677246094, 0.0005466341972351074, 3.5762786865234375e-06, 6.532669067382812e-05, 0.0016385912895202637, 0.9999994039535522, 0.0060122013092041016, 0.999976634979248, 1.0, 0.004826724529266357, 8.58306884765625e-06, 0.04228854179382324, 0.04090845584869385, 0.0052075982093811035, 0.05548298358917236, 0.7747512459754944, 0.007404744625091553, 0.8778734803199768, 0.335885226726532, 0.014220952987670898, 0.9984472393989563, 1.0, 0.9916241765022278, 0.00030994415283203125]

# classification_confidences = [2.2649765014648438e-06, 0.0, 0.999915599822998, 0.9553214311599731, 1.0, 0.06961023807525635, 0.998100221157074, 0.9999967813491821, 0.0, 1.0, 0.9948939085006714, 0.9959185719490051, 0.9635053277015686, 0.0005009174346923828, 0.0, 1.1920928955078125e-07, 0.0, 0.9999991655349731, 0.9999880790710449, 0.0, 1.0, 0.0, 1.0, 0.9661983847618103, 1.0, 1.0, 0.0, 0.9999879598617554, 0.996575653553009, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.9900248050689697, 0.06685954332351685, 0.9331404566764832, 0.011318087577819824, 0.9999998807907104, 0.7912163138389587, 0.0, 1.0, 0.7160517573356628, 1.0, 1.0, 5.4836273193359375e-06, 1.0, 1.0, 1.0, 0.8579521179199219, 0.9999998807907104, 1.0, 2.5987625122070312e-05, 2.0742416381835938e-05, 0.9101178646087646, 1.0, 1.7881393432617188e-06, 0.9999982118606567, 0.9981783628463745, 0.9999876022338867, 0.9999876022338867, 0.9826831817626953, 0.9999922513961792, 0.9985602498054504, 1.0, 0.18874430656433105, 0.9999866485595703, 0.9999996423721313, 1.0, 0.9998999834060669, 0.496573805809021, 0.9999054670333862, 1.0, 1.0, 5.841255187988281e-06, 1.0, 1.0, 0.99998939037323, 1.0, 1.0, 0.9999790191650391, 1.0, 1.0, 1.0, 0.868572473526001, 0.07859188318252563, 1.0, 1.0, 0.0, 1.0, 0.996605634689331, 0.02990192174911499, 1.1920928955078125e-07, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999819815158844, 1.0, 2.2649765014648438e-06, 0.9706921577453613, 1.0, 0.9999759197235107, 0.41585272550582886, 0.3331642150878906, 0.9997144341468811]

classification_confidences = [0.015773475170135498, 0.07130545377731323, 0.019818484783172607, 0.962357223033905, 0.034673869609832764, 0.25371116399765015, 0.36840343475341797, 0.3241921663284302, 0.11601895093917847, 0.9903922080993652, 0.05293929576873779, 0.14574533700942993, 0.8506538271903992, 0.9611755013465881, 0.04970484972000122, 0.47145843505859375, 0.05728459358215332, 0.6948206424713135, 0.06174129247665405, 0.11400920152664185, 0.999515175819397, 0.9755890369415283, 0.39884233474731445, 0.5263242125511169, 0.761021077632904, 0.8497596383094788, 0.8043842315673828, 0.2030041217803955, 0.6271600127220154, 0.9529975056648254, 0.9921843409538269, 0.9921843409538269, 0.8883306980133057, 0.045295119285583496, 0.0898239016532898, 0.775842547416687, 0.15214771032333374, 0.05672144889831543, 0.9432785511016846, 0.06589430570602417, 0.2877747416496277, 0.09999752044677734, 0.091441810131073, 0.8985528349876404, 0.0678941011428833, 0.17281532287597656, 0.1061435341835022, 0.25244784355163574, 0.7490467429161072, 0.9825536608695984, 0.9161165356636047, 0.9902258515357971, 0.4522015452384949, 0.5628249645233154, 0.05019193887710571, 0.25507742166519165, 0.42385298013687134, 0.6999795436859131, 0.03353440761566162, 0.9664655923843384, 0.12346184253692627, 0.10396939516067505, 0.896030604839325, 0.017037272453308105, 0.9833870530128479, 0.999880313873291, 0.3334067463874817, 0.4709792733192444, 0.17157191038131714, 0.11182856559753418, 0.28204071521759033, 0.46995091438293457, 0.12130963802337646, 0.13278645277023315, 0.6894485950469971, 0.3022981286048889, 0.22616368532180786, 0.9912678599357605, 0.9989684820175171, 0.08870106935501099, 0.20343279838562012, 0.8510668277740479, 0.9411978125572205, 0.0134354829788208, 0.9485254287719727, 0.9875911474227905, 0.8790122270584106, 0.8448004126548767, 0.9998587369918823, 0.8372476100921631, 0.4518066644668579, 0.5750121474266052, 0.34188032150268555, 0.9859522581100464, 0.7115268111228943, 0.9954015016555786, 0.09813946485519409, 0.9018605351448059, 0.09813946485519409, 0.979454517364502, 0.6359785795211792, 0.11243832111358643, 0.3765091300010681, 0.819725751876831, 0.44382452964782715, 0.5403338670730591, 0.303769588470459, 0.27134114503860474, 0.9654706716537476]

# 50 normal, 50 tumor
# correct = [0 for i in range(0, 50)] + [ 1 for i in range(0, 50)]
# 45 normal, 64 tumor
correct = [0 for i in range(0, 45)] + [1 for i in range(0, 64)]

m = tf.keras.metrics.AUC(num_thresholds=100)
m.update_state(correct, classification_confidences)
print(m.result().numpy())
