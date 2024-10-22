[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py


## Simple
<img width="535" alt="image" src="https://github.com/user-attachments/assets/43bae74e-5743-4097-9187-6c1b4aee9e00">

```
Epoch  10  loss  34.07600725955134 correct 29  
Epoch  20  loss  34.01654852296385 correct 29  
Epoch  30  loss  33.95048356493237 correct 29  
Epoch  40  loss  33.84883832013189 correct 29  
Epoch  50  loss  33.648384745256394 correct 29  
Epoch  60  loss  33.171023605374266 correct 29  
Epoch  70  loss  31.686123750148013 correct 29  
Epoch  80  loss  27.14381650721197 correct 30  
Epoch  90  loss  17.654657882838 correct 45  
Epoch  100  loss  11.775944782239204 correct 50  
Epoch  110  loss  8.454993499179208 correct 50  
Epoch  120  loss  6.498747821313709 correct 50  
Epoch  130  loss  5.21897913506358 correct 50  
Epoch  140  loss  4.326647193602058 correct 50  
Epoch  150  loss  3.69320020179495 correct 50  
Epoch  160  loss  3.2110983246522484 correct 50  
Epoch  170  loss  2.8386076059632077 correct 50  
Epoch  180  loss  2.53762963905932 correct 50  
Epoch  190  loss  2.289307827328205 correct 50  
Epoch  200  loss  2.081231407121544 correct 50  
Epoch  210  loss  1.9047202726894799 correct 50  
Epoch  220  loss  1.753429373059681 correct 50  
Epoch  230  loss  1.6225782008015757 correct 50  
Epoch  240  loss  1.5084877125398553 correct 50  
Epoch  250  loss  1.4082813684802888 correct 50  
Epoch  260  loss  1.3196813611996945 correct 50  
Epoch  270  loss  1.2408643596692839 correct 50  
Epoch  280  loss  1.1703567001569086 correct 50  
Epoch  290  loss  1.1069568054246743 correct 50  
Epoch  300  loss  1.0527224668270203 correct 50  
Epoch  310  loss  1.000762498135507 correct 50  
Epoch  320  loss  0.9533358961631597 correct 50  
Epoch  330  loss  0.9099597013386259 correct 50  
Epoch  340  loss  0.8701457539740703 correct 50  
Epoch  350  loss  0.8374940641109403 correct 50  
Epoch  360  loss  0.8035597545901914 correct 50  
Epoch  370  loss  0.7721269612305744 correct 50  
Epoch  380  loss  0.7418881413458428 correct 50  
Epoch  390  loss  0.7159515874191287 correct 50  
Epoch  400  loss  0.6917842123062086 correct 50  
Epoch  410  loss  0.669117386056115 correct 50  
Epoch  420  loss  0.6472028380358272 correct 50  
Epoch  430  loss  0.6272161651957678 correct 50  
Epoch  440  loss  0.6083557529285262 correct 50  
Epoch  450  loss  0.5905650660581621 correct 50  
Epoch  460  loss  0.5737450299241649 correct 50  
Epoch  470  loss  0.5578133192026693 correct 50  
Epoch  480  loss  0.542697689128224 correct 50  
Epoch  490  loss  0.5283342920894756 correct 50  
Epoch  500  loss  0.5146663551540681 correct 50
```
## Diag  
<img width="478" alt="image" src="https://github.com/user-attachments/assets/5680e341-9876-428b-b65d-3213aefcea64">

```
Epoch  10  loss  15.410677162784587 correct 46  
Epoch  20  loss  13.757496532637692 correct 46  
Epoch  30  loss  13.49586215929452 correct 46  
Epoch  40  loss  13.3017680269251 correct 46  
Epoch  50  loss  13.099724717410215 correct 46  
Epoch  60  loss  12.871475509809658 correct 46  
Epoch  70  loss  12.600817539487412 correct 46  
Epoch  80  loss  12.285077991472788 correct 46  
Epoch  90  loss  11.908798909472726 correct 46  
Epoch  100  loss  11.457218919570971 correct 46  
Epoch  110  loss  10.919142695906162 correct 46  
Epoch  120  loss  10.290095970570878 correct 46  
Epoch  130  loss  9.571840560944002 correct 46  
Epoch  140  loss  8.782519260186803 correct 46  
Epoch  150  loss  7.9504692842724385 correct 46  
Epoch  160  loss  7.12282050637939 correct 46  
Epoch  170  loss  6.329813499486038 correct 48  
Epoch  180  loss  5.599599825534338 correct 48  
Epoch  190  loss  4.954947186908314 correct 48  
Epoch  200  loss  4.4045355531299615 correct 48  
Epoch  210  loss  3.9928667559790902 correct 48  
Epoch  220  loss  3.689639746950225 correct 48  
Epoch  230  loss  3.4311904247950307 correct 48  
Epoch  240  loss  3.1912631187995717 correct 49  
Epoch  250  loss  2.9682546270195505 correct 49  
Epoch  260  loss  2.760767199654486 correct 49  
Epoch  270  loss  2.5683601625114165 correct 49  
Epoch  280  loss  2.3912422376342874 correct 49  
Epoch  290  loss  2.227786272345671 correct 50  
Epoch  300  loss  2.0772124162878436 correct 50  
Epoch  310  loss  1.9387419733191764 correct 50  
Epoch  320  loss  1.8115473115442007 correct 50  
Epoch  330  loss  1.6948250136146918 correct 50  
Epoch  340  loss  1.587766770732335 correct 50  
Epoch  350  loss  1.4897092326191197 correct 50  
Epoch  360  loss  1.3999532579753606 correct 50  
Epoch  370  loss  1.3175158995529104 correct 50  
Epoch  380  loss  1.241745931341367 correct 50  
Epoch  390  loss  1.172035704151841 correct 50  
Epoch  400  loss  1.107840177993398 correct 50  
Epoch  410  loss  1.048648332799484 correct 50  
Epoch  420  loss  0.9940045793286945 correct 50  
Epoch  430  loss  0.9435344615642235 correct 50  
Epoch  440  loss  0.8969575506653314 correct 50  
Epoch  450  loss  0.8537568030179876 correct 50  
Epoch  460  loss  0.8136303375162547 correct 50  
Epoch  470  loss  0.7763121887147432 correct 50  
Epoch  480  loss  0.7415547126543659 correct 50  
Epoch  490  loss  0.7091395431994746 correct 50  
Epoch  500  loss  0.6793709055727688 correct 50
```
## Split
<img width="554" alt="image" src="https://github.com/user-attachments/assets/2d4148ea-baa4-4cbf-9790-d38b5c80e460">

```
Epoch  10  loss  33.58685093659868 correct 27  
Epoch  20  loss  32.44350067237807 correct 27  
Epoch  30  loss  31.550856967379467 correct 32  
Epoch  40  loss  30.31426811813752 correct 40  
Epoch  50  loss  28.87059328830488 correct 44  
Epoch  60  loss  27.244473349466304 correct 44  
Epoch  70  loss  25.532181227684454 correct 45  
Epoch  80  loss  23.63005876029131 correct 45  
Epoch  90  loss  21.626300001131145 correct 46  
Epoch  100  loss  19.519420844969282 correct 47  
Epoch  110  loss  17.435238933471773 correct 47  
Epoch  120  loss  15.456052086942238 correct 48  
Epoch  130  loss  13.649955394496578 correct 48  
Epoch  140  loss  12.078413002958827 correct 48  
Epoch  150  loss  10.731279391997578 correct 49  
Epoch  160  loss  9.586662122240387 correct 49  
Epoch  170  loss  8.58421904016134 correct 49  
Epoch  180  loss  7.725434742116548 correct 49  
Epoch  190  loss  6.936047093185877 correct 49  
Epoch  200  loss  6.120442825818734 correct 49  
Epoch  210  loss  5.510360553887923 correct 49  
Epoch  220  loss  5.081564665696439 correct 49  
Epoch  230  loss  4.727179187921648 correct 49  
Epoch  240  loss  4.419657332660831 correct 50  
Epoch  250  loss  4.147386859730348 correct 50  
Epoch  260  loss  3.9047946213178952 correct 50  
Epoch  270  loss  3.6874007549066596 correct 50  
Epoch  280  loss  3.4914495881451293 correct 50  
Epoch  290  loss  3.3140134203412552 correct 50  
Epoch  300  loss  3.1525097514442577 correct 50  
Epoch  310  loss  3.0049853093245837 correct 50  
Epoch  320  loss  2.869699625051317 correct 50  
Epoch  330  loss  2.745183835171273 correct 50  
Epoch  340  loss  2.6302800403199695 correct 50  
Epoch  350  loss  2.523944485937836 correct 50  
Epoch  360  loss  2.4252774043814487 correct 50  
Epoch  370  loss  2.333502911938961 correct 50  
Epoch  380  loss  2.247937964602424 correct 50  
Epoch  390  loss  2.167987949167168 correct 50  
Epoch  400  loss  2.093051574858764 correct 50  
Epoch  410  loss  2.022596793766591 correct 50  
Epoch  420  loss  1.9563972347623322 correct 50  
Epoch  430  loss  1.893993612786997 correct 50  
Epoch  440  loss  1.8350559211949564 correct 50  
Epoch  450  loss  1.7794197480619258 correct 50  
Epoch  460  loss  1.726684915741344 correct 50  
Epoch  470  loss  1.6758125879189913 correct 50  
Epoch  480  loss  1.6276080990875854 correct 50  
Epoch  490  loss  1.581888038798674 correct 50  
Epoch  500  loss  1.5384431160635774 correct 50  
Epoch  510  loss  1.4970843695224807 correct 50  
Epoch  520  loss  1.4577137981018438 correct 50  
Epoch  530  loss  1.4201916257440226 correct 50  
Epoch  540  loss  1.3843914714555323 correct 50  
Epoch  550  loss  1.3501995720737634 correct 50  
Epoch  560  loss  1.3175111258087593 correct 50  
Epoch  570  loss  1.2856081990798027 correct 50  
Epoch  580  loss  1.2547337284403357 correct 50  
Epoch  590  loss  1.2264048203893794 correct 50  
Epoch  600  loss  1.199221834455786 correct 50  
Epoch  610  loss  1.1731160658429987 correct 50  
Epoch  620  loss  1.1482111241934128 correct 50  
Epoch  630  loss  1.1243906367346916 correct 50  
Epoch  640  loss  1.1014499267232842 correct 50  
Epoch  650  loss  1.0793457120559309 correct 50  
Epoch  660  loss  1.0580353461531589 correct 50  
Epoch  670  loss  1.0374662114287263 correct 50  
Epoch  680  loss  1.0176016416511136 correct 50  
Epoch  690  loss  0.9984055962258275 correct 50  
Epoch  700  loss  0.9798439473472168 correct 50
```

## XOR  
<img width="452" alt="image" src="https://github.com/user-attachments/assets/1828746b-9db4-405c-8f38-e421e54ae05e">

```
Epoch  10  loss  31.379544051341398 correct 32  
Epoch  20  loss  30.788466970541254 correct 32  
Epoch  30  loss  30.070541952325495 correct 33  
Epoch  40  loss  29.156071434323543 correct 36  
Epoch  50  loss  27.8881078519862 correct 37  
Epoch  60  loss  26.620542709678862 correct 39  
Epoch  70  loss  25.65895183489174 correct 39  
Epoch  80  loss  26.76584669739771 correct 39  
Epoch  90  loss  25.96523268610446 correct 39  
Epoch  100  loss  23.625961125936954 correct 38  
Epoch  110  loss  22.24078811118772 correct 39  
Epoch  120  loss  23.089882769756354 correct 45  
Epoch  130  loss  22.9815876755264 correct 41  
Epoch  140  loss  20.8226826984502 correct 45  
Epoch  150  loss  18.631661230875643 correct 45  
Epoch  160  loss  17.495741426941468 correct 46  
Epoch  170  loss  17.028468277932728 correct 45  
Epoch  180  loss  14.013630767567642 correct 47  
Epoch  190  loss  13.579727619188914 correct 46  
Epoch  200  loss  13.721756263229445 correct 46  
Epoch  210  loss  13.390306208348704 correct 45  
Epoch  220  loss  11.311719498921311 correct 46  
Epoch  230  loss  9.517750346474289 correct 48  
Epoch  240  loss  11.625807212804375 correct 45  
Epoch  250  loss  6.639425575359732 correct 49  
Epoch  260  loss  6.120713357743954 correct 49  
Epoch  270  loss  8.594419903773634 correct 47  
Epoch  280  loss  14.675382276820395 correct 43  
Epoch  290  loss  5.481351301630991 correct 49  
Epoch  300  loss  4.9970596058097465 correct 49  
Epoch  310  loss  4.606098936420791 correct 50  
Epoch  320  loss  4.304966136354351 correct 50  
Epoch  330  loss  7.711773229944429 correct 47  
Epoch  340  loss  10.574747709023935 correct 45  
Epoch  350  loss  4.098697509956942 correct 50  
Epoch  360  loss  3.771637032166699 correct 50  
Epoch  370  loss  3.527311969460251 correct 50  
Epoch  380  loss  3.3566754184723138 correct 50  
Epoch  390  loss  3.5519991055690485 correct 49  
Epoch  400  loss  3.7332408951971265 correct 48  
Epoch  410  loss  4.582801918386094 correct 48  
Epoch  420  loss  6.610353047816417 correct 47  
Epoch  430  loss  49.52336855078852 correct 36  
Epoch  440  loss  3.19608034036609 correct 50  
Epoch  450  loss  2.971428628016894 correct 50  
Epoch  460  loss  2.810630288785993 correct 50  
Epoch  470  loss  2.673475812352413 correct 50  
Epoch  480  loss  2.563678239873114 correct 50  
Epoch  490  loss  2.427910567641699 correct 50  
Epoch  500  loss  2.344309859414105 correct 50
```
