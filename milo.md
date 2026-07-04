MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient
Surface Reconstruction
ANTOINEGU╔DON? andDIEGOGOMEZ?,╔colePolytechnique,France
NISSIMMARUANI,Inria,UniversitΘC⌠tedÆAzur,France
BINGCHENGONG,╔colePolytechnique,France
GEORGEDRETTAKIS,Inria,UniversitΘC⌠tedÆAzur,France
MAKSOVSJANIKOV,╔colePolytechnique,France
Ours - Mesh Ours - Render Ours - Render Ours - Mesh
Ours - 302 MB RaDe-GS - 2.2 GB Ours - F1? 0.76 GOF - F1? 0.68
Fig.1. Mesh-in-the-LoopGaussianSplatting.Ourmethodintroducesanoveldifferentiablemeshextractionframeworkthatoperatesduringtheoptimization
of3DGaussianSplattingrepresentations.Ateverytrainingiteration,wedifferentiablyextractameshùincludingbothvertexlocationsandconnectivityù
directlyfromGaussianparameters.ThisenablesgradientflowfromthemeshtoGaussians,allowingustopromotebidirectionalconsistencybetween
volumetric(Gaussians)andsurface(extractedmesh)representations.ThisapproachguidesGaussianstowardconfigurationsbettersuitedforsurface
reconstruction,resultinginhigherqualitymesheswithsignificantlyfewervertices.Inthisexample,ourmethodreconstructstheentirebicyclesceneùincluding
backgroundùwithalmost10timesfewerverticesthanpreviousmethodswhilepreservingfinegeometricdetails.Ourframeworkcanbepluggedintoany
Gaussiansplattingrepresentation,increasingperformancewhilegeneratinganorderofmagnitudefewermeshvertices.MILomakesthereconstructionsmore
practicalfordownstreamapplicationslikephysicssimulationsandanimation.
WhilerecentadvancesinGaussianSplattinghaveenabledfastreconstruc- meshesremainsachallenge.Currentapproachesextractthesurfacethrough
tionofhigh-quality3Dscenesfromimages,extractingaccuratesurface costlypost-processingsteps,resultinginthelossoffinegeometricdetailsor
requiringsignificanttimeandleadingtoverydensemesheswithmillionsof
?Bothauthorscontributedequallytothepaper. vertices.Morefundamentally,theaposterioriconversionfromavolumetric
toasurfacerepresentationlimitstheabilityofthefinalmeshtopreserve
AuthorsÆ Contact Information: Antoine GuΘdon, antoine.guedon@enpc.fr; Diego allgeometricstructurescapturedduringtraining.WepresentMILo,anovel
Gomez,╔colePolytechnique,France,diego.gomez@polytechnique.edu;NissimMaru-
GaussianSplattingframeworkthatbridgesthegapbetweenvolumetricand
ani,Inria,UniversitΘC⌠tedÆAzur,France,nissim.maruani@inria.fr;BingchenGong,
╔colePolytechnique,France,bingchen.gong@polytechnique.edu;GeorgeDrettakis, surfacerepresentationsbydifferentiablyextractingameshfromthe3D
Inria,UniversitΘC⌠tedÆAzur,France,George.Drettakis@inria.fr;MaksOvsjanikov, Gaussians.Wedesignafullydifferentiableprocedurethatconstructsthe
╔colePolytechnique,France,maks@lix.polytechnique.fr. meshùincludingbothvertexlocationsandconnectivityùateveryiteration
directlyfromtheparametersoftheGaussians,whicharetheonlyquantities
Permissiontomakedigitalorhardcopiesofallorpartofthisworkforpersonalor optimizedduringtraining.
classroomuseisgrantedwithoutfeeprovidedthatcopiesarenotmadeordistributed
Ourmethodintroducesthreekeytechnicalcontributions:(1)abidirec-
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation
onthefirstpage.Copyrightsforcomponentsofthisworkownedbyothersthanthe tionalconsistencyframeworkensuringbothrepresentationsùGaussians
author(s)mustbehonored.Abstractingwithcreditispermitted.Tocopyotherwise,or andtheextractedmeshùcapturethesameunderlyinggeometryduring
republish,topostonserversortoredistributetolists,requirespriorspecificpermission training;(2)anadaptivemeshextractionprocessperformedateachtraining
and/orafee.Requestpermissionsfrompermissions@acm.org.
iteration,whichusesGaussiansasdifferentiablepivotsforDelaunaytriangu-
⌐2025Copyrightheldbytheowner/author(s).PublicationrightslicensedtoACM.
ACM1557-7368/2025/12-ART1 lation;(3)anovelmethodforcomputingsigneddistancevaluesfromthe3D
https://doi.org/10.1145/3763339
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.
5202
tcO
92
]VC.sc[
2v69042.6052:viXra

1:2 ò AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,GeorgeDrettakis,andMaksOvsjanikov
Gaussiansthatenablesprecisesurfaceextractionwhileavoidinggeometric Moreover,manyexistingmethods[Huangetal.2024;Wangetal.
erosion. 2021;Yarivetal.2021;Zhangetal.2024a]focusonobject-centric
Ourapproachcanreconstructcompletescenes,includingbackgrounds, reconstructions,evaluatingonlythequalityoftheforegroundgeom-
withstate-of-the-artqualitywhilerequiringanorderofmagnitudefewer etry.Similarly,benchmarkslikeDTU[Jensenetal.2014]orTanks
meshverticesthanpreviousmethods.
andTemples[Knapitschetal.2017]offergroundtruthsolelyfor
Duetotheirlightweightandemptyinterior,ourmeshesarewellsuited
foregroundobjects,discouragingfull-scenereconstructionsthat
fordownstreamapplicationssuchasphysicssimulationsandanimation.
havebackgrounds.
Thecodeforourapproachandanonlinegalleryareavailableathttps:
Inthiswork,weintroduceanovelpipelineforreconstructing
//anttwo.github.io/milo/.
compact,high-fidelitymeshesofcomplete3DscenesusingGaussian
CCSConcepts:òComputingmethodologies?Reconstruction;Mesh Splatting[Kerbletal.2023],addressingalloftheaforementioned
models;Point-basedmodels;Shaperepresentations;Rendering;Image-based limitations.Ourcoreideaistogenerateameshateverytraining
rendering. iterationfromasetofpointsentangledwiththeGaussians,which
wecallGaussianPivots.
AdditionalKeyWordsandPhrases:Mesh,GaussianSplatting
Thissetupenablesdirectgradientbackpropagationfromthemesh
ACMReferenceFormat: totheGaussianparameters,allowingustoimposeconsistencybe-
AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,George tweenthesurfacemesh,volumetricfield,andinputdata.Thanksto
Drettakis,andMaksOvsjanikov.2025.MILo:Mesh-In-the-LoopGaussian
thePivots,Gaussiansserveasanimplicitparameterizationofanex-
SplattingforDetailedandEfficientSurfaceReconstruction.ACMTrans.
plicitmesh,progressivelyoptimizedtoyieldhigh-qualitygeometry,
Graph.44,6,Article1(December2025),15pages.https://doi.org/10.1145/
vertexpositions,andconnectivity.
3763339
Asaresult,ourapproachsidestepsfixed-topologyconstraints,
giventhattheconnectivityisdynamicallyupdatedasGaussians
1 Introduction
movearoundthescene.Ontheotherhand,themeshprovidesa
Recentmethods[Chenetal.2023;GuΘdonandLepetit2024b;Huang geometricpriorthatregularizesGaussianstoaddressôcheating",
etal.2024;Lietal.2023;Reiseretal.2024;Yuetal.2024;Zhangetal. mitigatingundesirablegeometricartifacts.Thistight,bidirectional
2024]forreconstructingsurfacemeshesfromimagescommonly couplingenableseffectivemesh-basedregularization,enhancing
adoptatwo-stagepipeline:Optimizeavolumetricrepresentation bothfidelityandcompactness.Inotherwords,bothrepresenta-
viadifferentiablerenderingùtypicallyusingNeuralRadianceFields tionshelpeachother.
(NeRFs) or 3D Gaussian Splattingùthen extract a surface mesh Tofurthercontrolthecomplexityofoutputs,weproposenovel
duringpostprocessingbydefiningasurface,usuallyasanisosurface. densificationandsimplificationstrategiesinspiredby[Fangand
Noneofthesemethodsconsidersthereconstructedmeshduring Wang2024b].
optimization. We achieve state-of-the-art geometric quality while using an
This strategy offers no guarantee that the final mesh will be orderofmagnitudefewerverticesthancompetingmethods,mak-
consistentwiththevolumetricrepresentationùfinedetailsthatare ingoursurfacesfarmorepracticalfordownstreamapplications.
presentwhenrenderingtherepresentationcanbelost. Inspiredinpreviousworkswhichfocusonsurface-basedviewsyn-
Thispostprocessingstepcollapsescomplexvolumetricinforma- thesismethods[Reiseretal.2024]weuseaproxymetricbasedon
tionintoasurface,oftenleadingtothelossoffineorsemitransparent meshrenderingtoevaluatemeshqualityinbackgroundregions
structures,andcanintroduceartifactsintheextractedmesh.More- wherenoground-truthgeometryexists,traininganeuralcolorfield
over,GaussianSplattingandNeRF-basedmethodsareknownto overthemeshandcomparingrenderedimagesacrossmethods.This
adjusttheiropacityandview-dependentcolorsindependentlyofthe decouplingrenderingfidelityfrommeshresolutionandhelpsusto
geometry.Thisallowsthemtofitthetrainingimagesmoreprecisely, evaluatereconstructionqualityacrosstheentirescene.Ourwork
butoftenattheexpenseofgeometricconsistency.ôCheating"leads makesthefollowingcontributions.
tohallucinatedstructuressuchasfloatersorcavities,whichare
particularlyhardtoresolveduringmeshextraction.Fixingthemis ò Weintroducethefirstradiancefieldpipelineinwhichextract-
non-trivial,astheunderlyingvolumetricrepresentationhasalready ingasurfacemeshisanintegralpartoftheoptimization,lever-
absorbedtheseinconsistenciesinitsoptimization.Severalworks agingexpressive3DGaussianSplattingtoparameterizeand
havehighlightedthisissueinchallengingsettingsùsuchaslow- jointlyrefinebothrepresentations.
information[Gomezetal.2025;GuΘdonetal.2025;Warburgetal. ò We propose mesh-based regularization strategies that im-
2023;Yangetal.2023]orhighspecularities[Verbinetal.2022]ùyet provegeometricquality,especiallyforthinstructures.
ageneralandrobustsolutionremainselusive. ò Ourmethodachievesstate-of-the-artresultsinmeshquality
Crucially,thisprocessprovidesnofeedbackduringoptimization, andcompactnessacrossmultiplecomplex3Dscenes,improv-
leaving mesh quality undefined until training is complete. This inguponpreviousapproachesinbothscalabilityandvisual
meansthatthereisnoguaranteethattheoptimizedGaussianrepre- fidelity.
sentationwillgiverisetoahighqualitymeshcapturingallrelevant ò Weadaptametricintroducedinthecontextofsurface-based
structures.Forinstance,finedetailssuchasbicyclespokescandisap- viewsynthesisanduseitasanevaluationprotocoltoassess
pearorbeinaccuratelyrepresentedinthefinalmeshevenifvisually full-scene geometry, even in the absence of ground-truth
theyexistinthevolumetricrepresentation. 3D models. We use this protocol to demonstrate that our
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

MILo:Mesh-In-the-LoopGaussianSplattingforDetailedandEfficientSurfaceReconstruction ò 1:3
methodachievesstate-of-the-artresultsinmeshqualityand Moreover,thesevolumetricrepresentationstypicallycannotachieve
compactnessacrossmultiplecomplex3Dscenes, real-timerendering,limitingtheirpracticalapplications.Someap-
proachesaddressthisbyfirstoptimizingthevolumetricrepresenta-
We provide detailed definitions of 3D Gaussian Splatting and tion,extractingamesh,andthenbakingtherenderingcapabilities
DelaunayTriangulationinthesupplementarymaterial. intothemeshtoenablereal-timerendering[Reiseretal.2024;Yariv
etal.2023].Thus,thesemethodsaresplitintotwostages,whichpre-
ventsanyrefinementofthemesh,whilestillexhibitinglengthy
2 RelatedWork optimizationtimes.
Othermethodsbasedondiscretestructures[Munkbergetal.2022;
Novel View Synthesis. Early novel view synthesis (NVS) tech-
Shenetal.2023]alsoprovedefficienttosomeextent,butfocuson
niques modeled scenes as continuous volumetric fields, notably
individualobjectsandarenotscalabletolargecomplexscenes.
withNeRF[Mildenhalletal.2020]introducingamultilayerpercep-
RecentmethodsrelyingonGaussianSplattingofferpromising
trontolearnaradiancefield.Subsequentworksexploredalternative
efficient alternatives [Dai et al. 2024; Wang et al. 2023] namely:
representations,includinghash-basedencodings[Mⁿlleretal.2022],
2DGS[Huangetal.2024],RaDe-GS[Zhangetal.2024],NeuSG[Chen
discretevoxelgrids[Fridovich-Keiletal.2022],andlow-ranktensor
etal.2023],VCR-GauS[Chenetal.2024],andQuadraticGaussian
decompositions[Chenetal.2022],aswellassolutionsfocusedon
Splatting[Zhangetal.2024a]exploredifferentrenderingorregu-
antialiasing[Barronetal.2021;Barronetal.2022,2023].
larizationstrategiesthatrelyonTruncatedSignedDistanceFields
Recently,3DGaussianSplatting[Kerbletal.2023]hasredefined
(TSDF),whileSuGaR[GuΘdonandLepetit2024b]andGaussian
theNVSlandscapebyenablingfaster,high-qualitysynthesisfrom
OpacityFields[Yuetal.2024]proposetheirownmeshextraction
point-basedprimitives.Thissparkedawaveoffollow-upworkaim-
procedures.
ingtoimproveefficiencyandscalability,includingMip-Splatting[Yu
All previously highlighted methods perform mesh extraction
etal.2024a],Mini-Splattingv1andv2[FangandWang2024a,b],
after optimization,treatingitasapost-processingstep.Someof
MCMC-Gaussians[Kheradmandetal.2024],BetaSplatting[Liuetal.
thesegoslightlyfurther,adoptingatwo-stageapproachwherethey
2025],andTaming3DGS[Mallicketal.2024].Additionalworklike
firstoptimizeavolumetricrepresentation,extractamesh,andthen
3DGUT[Wuetal.2024]and3DGRT[Moenne-Loccozetal.2024]
refineitwithdifferentiablerendering[GuΘdonandLepetit2024a,b;
extendthesemodelstoarbitrarycamerasystemsanddifferentiable
Yarivetal.2023].Thislineofwork,however,neitheravoidsthe
raytracing,respectively.
initialextractionissuesnoradjuststhemeshtopology,which
Despitetheirimpressiverenderingperformance,noneofthese
staysfixedduringtherefinementprocess.Thisseparation,inboth
methodsaimatexplicitsurfacemeshreconstruction,limitingtheir
cases,introducespotentialinconsistencies:thereisnoguarantee
applicabilityintasksrequiringgeometricreasoningordownstream
thattheextractedsurfaceaccuratelyreflectstheunderlying
editing.
volumetricrepresentation.SinceGaussiansarecontinuousby
nature,naiveisosurfacingoftenresultsingeometricartifactssuchas
Surfacereconstructionfromimages. Recoveringexplicitsurface over-inflationorerosion.Theseartifactsareparticularlynoticeable
meshesexclusivelyfromimagesisalongstandingchallenge.Well- aroundthinstructures(seeFig.1).Moreover,allofthesemethods
establishedvolumetricapproachesrelyingonimplicitfunctions[Li typicallyproduceoverlydensemeshes(uptotensofmillionsof
etal.2023;Wangetal.2021;Yarivetal.2021]provedefficientfor vertices)thataredifficulttoscaletolargescenes.
reconstructingaccuratesurfaces.Traininganimplicitfunctionis Incontrast,ourmethodefficientlyleveragesGaussianstoscaleto
astrongregularizationthatallowstoobtainsmoothsurfaces,thus full-scenereconstruction,andisthefirsttointegratemeshextrac-
thesemethodsexcelatextractingtheforegroundofobject-centric tiondirectlyintotheoptimizationloop.Thisdesignnaturally
scenes. enablestherefinementofbothvertexpositionsandmeshtopology,
However, these methods generally require very long training ensuringconsistencywiththeunderlyingvolumetricrepresentation
times(oftenexceeding24hours)beforemeshextractioncanbeper- (Gaussiansplats).
formed.ThesuccessofGaussianSplattinghasnaturallyinspiredan Voronoi&Delaunay-basedmethods. Ourworkalsoexploitsthe
emerginglineofworkthattrainssplatsjointlywithimplicitfunc- VoronoidiagramandspecificallyitsdualtheDelaunaytriangula-
tions,notablyGaussian-UDF[Lietal.2025],GS-Pull[Zhangetal. tion,whichareclassicalgeometricconstructionswithdeeptheoret-
2024b],andGSDF[Yuetal.2024b].Bycombiningthesetworepre- icalandpracticalrelevance[Aurenhammer1991].Thesestructures
sentations,suchmethodsmitigatethetimebottleneckbyquerying havelongbeenappliedtosurfacereconstructionproblems[Amenta
theimplicitfunctionthroughstrategiesthatexploitthepositionsof etal.1998,2001;DeyandGoswami2003],withexplicitguarantees
theGaussiansplats.Whilethisimprovestrainingefficiency,italso underspecificsamplingconditions.Morerecently,theyhavealso
exacerbatesscalabilityissues,sincetworepresentationsmustbe gainedtractioninmachinelearningcontexts:in2Dvision[Williams
optimizedsimultaneously.Moreover,althoughneuralSDFsexcelat etal.2020],3Dgeometry[Maruanietal.2023,2024],orevennovel
reconstructingisolatedobjects,theirexpressivityremainsbounded viewsynthesis[Elsneretal.2023;Govindarajanetal.2025].While
bytheMLParchitecture[Lietal.2025;Yuetal.2024b;Zhangetal. GOF [Yu et al. 2024] applies Delaunay triangulation as a post-
2024b].Consequently,whiletheseapproachescaninprinciplebe processingstepformeshextraction,RadiantFoam[Govindarajan
appliedtomorecomplexscenes,theytendtounderperformbeyond etal.2025]incorporatesitintoitstrainingpipeline,butfocuses
single-objectorforeground-dominantsettings. solelyonviewsynthesisandnotsurfacereconstruction.Incontrast,
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

1:4 ò AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,GeorgeDrettakis,andMaksOvsjanikov
Fig.2. MILoPipeline.WeuseGaussiansaspivotsfortheparametrizationofameshwecandifferentiablyextractateveryiteration.Supervisiononthismesh
allowsustoimposeasoftpriorontheGaussianswhichresultsinbetterreconstructedsurfaces
ourworkisthefirsttoleverageDelaunaytriangulationin-the-loop
Table1. Resourcerequirementsfordifferentmethods.Wecomparethe
computationalresourcesrequiredfortrainingonhigh-resolutionimages,
withtheexplicitgoalofproducinghigh-qualitymeshes,enablingdi-
aswellastheresultingmodelsizesacrossdifferentapproaches.Allmea-
rectsupervisionofsurfacegeometrythroughouttraining.
surementsareaveragedacrossscenesfromtheTanks&Templesdataset.
Ourmethodachieveshigh-qualityreconstructionwithsignificantlylower
resourcerequirements,particularlyregardingmeshcomplexity.Thebase
versionofourmodelusesupto10xfewerGaussiansthancompetingmeth-
3 Overview
odsdependingonthescene,producingmesheswithfewerverticesand
Ourmethodaimstoaddressthefundamentalchallengeofextracting triangleswhilepreservingdetailsandachievingbetteraccuracy,makingit
high-qualitysurfacemeshesfrom3DGaussianSplattingrepresenta- moresuitablefordownstreamapplications.
tions.Ratherthantreatingmeshextractionasapost-processingstep,
TrainingResources OutputMesh
weintegrateitdirectlyintotheoptimizationprocessandensurethat #Gaussians(M) GPUMem(GB) Time #Vertices #Triangles Size(MB)
boththevolumetricandsurfacerepresentationsareconsistent.By 2DGS 0.98 4.7GiB 29m 16.39M 21.68M 557.1
GOF 1.55 10.6GiB 93m 16.49M 33.17M 600.0
extractingameshateverytrainingiterationandback-propagating RaDe-GS 1.56 12.4GiB 42m 14.75M 29.59M 592.0
gradientstotheGaussians,weguidethemtowardconfigurations Ours(base) 0.28 10.0GiB 50m 4.36M 8.97M 179.6
Ours(dense) 2.11 16.5GiB 110m 6.89M 13.79M 276.1
optimizedforaccuratesurfaceextraction.SeeFig.2foravisual
overviewofourmethod. 4 DifferentiableMeshExtraction
Ateachtrainingiteration,ourpipelineconsistsoffivemainsteps:
Inthissection,weprovidemoredetailsonthedifferentstepsofour
differentiablemeshextraction,allowinggradientstoflowfromthe
(1) FetchthetrainableDelaunayverticesderivedfromtheGauss-
verticesofthemeshbacktotheparametersoftheGaussiansduring
ianPivots(Sec.4.1).
optimization.
(2) UpdatetheDelaunaytriangulation(Sec.4.1).
(3) FetchthetrainablesigneddistancevaluesforeachDelaunay 4.1 SamplingDelaunayVerticesfromGaussians
vertex(Sec.4.2).
Thefirststepforextractingasurfacemeshistoidentifyappropriate
(4) ApplyGPU-baseddifferentiableMarchingTetrahedratoex-
pointsin3DspacethatwillserveastheverticesofourDelaunay
tractthemesh(Sec.4.3).
triangulation.Whileanaiveapproachwouldconsistinusingthe
(5) Back-propagateimage-basedandconsistencylossestothe
centersof allGaussiansasDelaunayvertices,thissimplestrategy
Gaussianparameters,bysimultaneouslyrenderingtheex-
presentstwosignificantlimitations:
tractedmeshandthe3DGaussians.(Sec.5).
(1) ThecentersofGaussiansaretypicallylocatedonornear
Duringoptimization,weenforcegeometricconsistencybetween thesurfaceofthescene,whereasforeffectiveapplicationof
themeshandtheGaussiansbyrenderingandcomparingdepthand marchingtetrahedra,oneneedsDelaunayverticesstraddling
normalmapswithbothrepresentations,encouragingtheextracted thetargetsurface.
surfacetomatchthegeometryencodedinthe3DGaussians.This (2) UsingallGaussiansasDelaunayverticescanbecomputa-
bidirectionalconsistencyframeworknotonlyimprovesthequality tionallyexpensiveforlargescenes.
oftheextractedmeshbutalsoencouragesGaussianstoconverge Toaddressthefirstissue,wefollowthestrategyproposedby
towardbetter,solidsurfaceswithmulti-viewconsistentgeometry. GaussianOpacityFields(GOF)[Yuetal.2024]:wesample9points
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

MILo:Mesh-In-the-LoopGaussianSplattingforDetailedandEfficientSurfaceReconstruction ò 1:5
perGaussian,includingitscenterand8cornerpointsalignedwith
itsprincipalaxes.Theseareobtainedbyscalingandrotatingthe
unitboundingboxverticesandcenter{? ,? ...? }:
0 1 8
? ?,? =? ? +? ? ╫(? ? ?? ?) for ? =0...8, (1)
where?istheHadamardproduct.
Thissamplingstrategyensuresthatthetriangulationadaptsto
theanisotropicnatureoftheprimitives,creatingamoreaccurate
representationofthesurface.
Toaddressthescalabilitychallenge,weproposetosampleDelau- Fig.3. OverviewofGaussianPivotsin2D.TwoGaussianseachgenerate
Delaunayvertices(blacksquares)withcontinuousSDFvalues(redand
nayverticesonlyfromGaussiansthataremostlikelytobelocated
bluedots).TheirDelaunaytriangulation(dashedlines)isiso-surfacedwith
nearthesurface.Todoso,weleveragetheimportance-weightedsam-
MarchingTetrahedra,whichplacesfinalmeshvertices(blackdots)onedges
plingintroducedinMini-Splatting2[FangandWang2024b],which
wheresignchangesoccur,resultingintheextractedmesh(boldlines).
ranksGaussiansbasedontheircontributiontorenderingacross
alltrainingviews,asreflectedbytheaveragemagnitudeoftheir
blendingcoefficientsalongcamerarays.Weusetheseimportance
thesevaluesdonotoriginatefromatrueSDF,werefertothemas
scoresassamplingprobabilities,allowingustoselectasubsetof
SDFvaluesforsimplicity.
Gaussiansthateffectivelypreservethegeometricstructureandfine
Weproposeasimpleyeteffectivestrategy:augmentingeach
detailsofthescene. GaussianG? with9optimizableSDFvalues? ? ? R9ùonefor
WhileonlyselectedGaussiansreceivedirectgradientupdates
eachDelaunayvertex.Importantly,theseSDFvaluesaredecoupled
fromthemeshregularization,sinceweimposeconsistencybetween
fromtheGaussiansÆotherparametersùsuchasopacity,scale,and
the 3DGS and mesh rendering (see section 5), all Gaussians are
rotationùallowingforlocalizedcontrolovertheextractedisosurface
constrainedinpractice.Theresultisacomputationallyefficientand
level.Thisdecouplingproveshighlybeneficialforaccuratelycap-
geometricallyaccuratetriangulation.
turingfinesurfacedetailsandensuringstrongconsistencybetween
Weproposetwovariantsofourmethod:
theextractedmeshandtheunderlyingvolumetricrepresentation.
ò Basemodel:WesampleasetofGaussiansusingImportance- Foranevenfasterconvergence,wedeviseacustominitializational-
weightedsampling[FangandWang2024b]andremoveall gorithm:pleaserefertothesupplementarymaterialformoredetails.
otherGaussiansfromthescene.Weextractameshateach AnillustrationofourapproachisprovidedinFig3.
iterationfromtheremainingGaussians.Thisresultsinalight-
4.3 DifferentiableMarchingTetrahedra
weightsetofGaussiansùbetween0.1Mand0.5Mdepending
onthecomplexityofthesceneùandacorrespondinglylight- OncetheDelaunayverticesandtheSDFvaluesarecomputed,we
weightmeshthatstillcapturesfinedetails. applyMarchingTetrahedra[DoiandKoide1991]toextractatri-
ò Dense model: We also sample a set of Gaussians using anglemesh.ForeachtetrahedronwithverticeshavingSDFvalues
Importance-weightedsampling[FangandWang2024b],but ofoppositesigns(indicatingthatthesurfaceintersectsthetetrahe-
wedonotremovetheotherGaussiansfromthescene:We dron),thealgorithmcomputestheintersectionpointsofthesurface
maintainthelargesetofGaussians(typicallybetween2M withtheedgesofthetetrahedron.These3(or4)intersectionpoints
and5M)forGaussianSplattingbutuseonlythesampledones formthevertices{? ?}ofthefinalmesh,andtheyaretriviallycon-
aspivotsforgeneratingtheDelaunayvertices.Thenumber nectedwith1(or2)trianglefaces.Specifically,giventwoDelaunay
ofDelaunayverticesisapproximatelythesameasthebase vertices?
?,?
and?
??,?
ofatetrahedronwithSDFvalues?
?,?
and?
??,?
model,stillresultinginalightweightmesh;however,having ofoppositesigns,thepositionof? ?isgivenby:
moreGaussianshelpsinlearningbetterSDFvaluesatthese
v b e e r t t t i e c r e p s. e T rf h o i r s m m a o n d c e e l . resultsinlongeroptimizationtimes,but ? ? = ? ?,? ? ? ? ? ? , , ? ? ? ? ? ? ? ? ? ? , , ? ? ? ?,? , (2)
OnceweobtaintheDelaunayvertices,wecomputetheirDelaunay Notethatthismeshextractionprocessallowsgradientstoflow
triangulationwhichprovidesatetrahedralizationrequiredforthe from the vertices of the extracted mesh back to the Gaussians
followingstep. through (a) the learnable SDF values and (b) the coordinates of
theDelaunayvertices,computedfromthemeansandcovariances
4.2 ComputingSignedDistanceValues oftheGaussians.
TocomputeasurfacemeshfromthepreviouslyobtainedDelaunay
5 Mesh-in-the-LoopOptimization
triangulation,werelyontheMarchingTetrahedraalgorithm[Doi
andKoide1991].Thisalgorithm,whichwewilldescribeinSec.4.3, Webuildontopofpreviousworks[Huangetal.2024;Kerbletal.
requiresatetrahedralgridaugmentedwithscalarvalues,typically 2023;Yuetal.2024;Zhangetal.2024],whoseoptimizationisbased
derivedfromaSignedDistanceField(SDF).Toapplyit,wemust onsuccessiveiterationsofrenderingthescenewith3DGaussians
assigna(signed)scalarvaluetoeachDelaunayvertex.Although andcomparingtheresultingimagetothetargetimagefromthe
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

1:6 ò AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,GeorgeDrettakis,andMaksOvsjanikov
Table2. QuantitativecomparisonontheTanks&TemplesDataset[Knapitschetal.2017].WereporttheF1-scoreandaverageoptimizationtime.All
resultsareevaluatedwiththeofficialevaluationscripts.Ours X usesthedifferentiablerasterizerfromX.VCR-GauS[Chenetal.2024]reliesonapre-trained
normalestimationmodel.Bestresultsforimplicitmethodsarehighlightedinblue;bestresultsforexplicitmethodsarehighlightedinred.Whenamethod
fails(OOM),wereportthemeanofthesuccessfulscenes,denotedbyasuperscriptasterisk(??).OurmethodachievesthebestF1scoreamongexplicit
representations,eventhoughitoutputslightermeshesthanotherapproaches.
Implicit Explicit
GaussianUDF GS-Pull GSDF NeuS Geo-Neus Neuralangelo SuGaR 3DGS 2DGS GOF RaDe-GS QGS VCR-GauS OursRaDe-GS(base) OursGOF(base) OursRaDe-GS(dense)
Barn 0.27 0.60 0.16 0.29 0.33 0.70 0.14 0.13 0.36 0.51 0.43 0.43 0.62 0.55 0.59 0.64
Caterpillar 0.17 0.37 0.11 0.29 0.26 0.36 0.16 0.08 0.23 0.41 0.32 0.31 0.26 0.38 0.39 0.38
Courthouse 0.03 0.16 OOM 0.17 0.12 0.28 0.08 0.09 0.13 0.28 0.21 0.26 0.19 0.28 0.29 0.31
Ignatius 0.38 0.71 0.34 0.83 0.72 0.89 0.33 0.04 0.44 0.68 0.69 0.79 0.61 0.73 0.78 0.76
Meetingroom 0.17 0.22 0.01 0.24 0.20 0.32 0.15 0.01 0.16 0.28 0.25 0.25 0.19 0.25 0.26 0.28
Truck 0.30 0.52 0.25 0.45 0.45 0.48 0.26 0.19 0.26 0.59 0.51 0.60 0.52 0.60 0.62 0.59
Mean 0.22 0.43 0.18? 0.38 0.35 0.50 0.19 0.09 0.30 0.46 0.40 0.44 0.40 0.47 0.49 0.49
Time 90m 37.6m 70m >24h >24h >24h 73m 7.9m 12.3m 69m 11.5m 120m 53m 50m 150m 110m
Fig.4. ExamplesofsurfacereconstructionresultsontheTanksandTemplesandMipNeRF360datasets.Ourmethodproducesmesheswithclean
surfacesandveryfinedetails,whilebeinglighterthanpreviousmethods(Stump:320MB,Garden:301MB,Barn:313MB)
5.2 Volume-to-SurfaceConsistency
captureddataset.OurapproachcanbepluggedintoanyGaussian-
basedmethodabletorenderdepthmapsandnormalmapsthrough
Toenforceconsistencybetweenthegeometryencodedin(a)the
differentiablerasterization[Huangetal.2024;Yuetal.2024;Zhang
Gaussiansasavolumetricrepresentationand(b)theextractedsur-
etal.2024].Pleasenotethatdependingonthemethod,thedefinition
facemesh,weintroducethefollowingloss:
oftherendereddepthandnormalmapsmaydiffer.
| Akeyadvantageofourapproachliesintheabilitytoextracta |     |     |     |     | =? +?           | .    |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --------------- | ---- | --- |
|                                                      |     |     |     | L   | mesh MD L MD MN | L MN | (5) |
trianglemeshduringoptimization,performanydifferentiableoper-
ationonthesurface,andbackpropagatetheresultinggradientsto HereL MD isadepthconsistencylossdefinedasfollows:
theGaussians.Tocouplethetworepresentations,werelyondif-
??
|     |     |     |     | L   | = log(1+|?(?)?? | (?)|) , |     |
| --- | --- | --- | --- | --- | --------------- | ------- | --- |
ferentiablemeshrendering:wedirectlycomparedepthandnormal MD M (6)
| mapsrenderedfrombothrepresentationstoenforcegeometriccon- |     |     |     |     | ?   |     |     |
| --------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
sistencybetweentheGaussiansandtheextractedsurface.Below,we with?thedepthmaprenderedfromtheGaussiansand? the
M
describethedifferentlossfunctionsinvolvedduringoptimization. depthmaprenderedfromthemesh.Thenormalconsistencyloss
L isdefinedas
MN
5.1 Volumetricrendering
|     |     |     |     |     | ??(cid:16) | (cid:17) |     |
| --- | --- | --- | --- | --- | ---------- | -------- | --- |
TooptimizetheGaussians,werelyonthesamevolumetricrendering L = 1?Nÿ(?)╖? (?) , (7)
|                                                      |     |     |     |     | MN M |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | ---- | --- | --- |
| lossL as[Huangetal.2024;Yuetal.2024;Zhangetal.2024], |     |     |     |     | ?    |     |     |
vol
with? (?)thenormalofthefaceofthemeshrasterizedatpixel?.
| whichconsistsoftwophotometrictermsandaregularizationterm: |                  |        |         | M              |     |     |     |
| --------------------------------------------------------- | ---------------- | ------ | ------- | -------------- | --- | --- | --- |
| L                                                         | =(1?? )L +? L    | +? L , | (3) 5.3 | Regularization |     |     |     |
| vol                                                       | RGB 1 RGB D-SSIM | N N    |         |                |     |     |     |
whereL istheL1loss,L isaD-SSIMtermandL isa Despitetheireffectiveness,thepreviouslossesarenotsufficientfor
| 1   | D-SSIM |     | N   |     |     |     |     |
| --- | ------ | --- | --- | --- | --- | --- | --- |
normalconsistencylossdefinedasthefollowingsumoverpixels?: reconstructingoptimalmeshes,andtwokeychallengesremain.
Erosion. Duringoptimization,ifallSDFvaluesinsideatetrahe-
|     | ??(cid:16)       | (cid:17) |                                                          |     |     |     |     |
| --- | ---------------- | -------- | -------------------------------------------------------- | --- | --- | --- | --- |
|     | L = 1?N(?)╖Nÿ(?) | ,        | (4)                                                      |     |     |     |     |
|     | N                |          | dronbecomepositive,finedetailsofthegeometrycanbeerodedor |     |     |     |     |
?
evenlost.Duetothesharpnatureofmeshrasterization,itbecomes
whereN? istheexpectednormalatpixel?computedfromvolu- extremelydifficultfortherepresentationtorecoverthisgeometry,
metricrendering,andNÿ ? isthenormaldirectionatpixel?obtained evenwithantialiasingappliedtomeshrenderings.Oncearegionis
byapplyingfinitedifferenceontherendereddepthmap,follow- eroded,thegradientsignalbecomesweakornon-existent,making
ing[Huangetal.2024;Yuetal.2024;Zhangetal.2024].Thisterm itchallengingforoptimizationtorestorethemissingsurface.
encouragesGaussianstoalignwiththeirneighborsanddrastically Toaddressthisissue,weproposearegularizationtermL erosion
reducesthenoiseintherendereddepthandnormalmaps. thatencouragesgeometrytobepreserved:
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

MILo:Mesh-In-the-LoopGaussianSplattingforDetailedandEfficientSurfaceReconstruction ò 1:7
6 Experiments
??
L erosion = max(0,? ?? ), (8) Inthissection,weevaluateourmethodonvariousdatasetsand
??? Del compareitwithstate-of-the-artapproachesforsurfacereconstruc-
where? isthesetofGaussianssampledfortheDelaunaytrian- tionfrommulti-viewimages.Wefirstdescribeourexperimental
Del
gulationand? ?? istheSDFvalueofthecenterofthe?-thGaussian. setup,thenpresentquantitativeandqualitativeresultsforsurface
AsmentionedinSection4,thecentersofallGaussianssampled reconstructionquality,followedbyablationstudiesandapplications
forthetriangulationareusedasDelaunayvertices:thisregular- inmesheditingandanimation.Weshowthattheobtainedmeshes
izationtermaimstoincludethecentersoftheseGaussiansinside areclean,accurateandlightasshowcasedinFig.4.
thesurfacebyencouragingtheirSDFvaluestobecomenegative,
ImplementationDetails. OurmethodisimplementedinPyTorch,
preventingerosioninallareaswhereatleastoneGaussianhasbeen
buildingupontheRaDe-GScodebaseforGaussianSplattingwith
sampled.L isappliedonlytothecentersofselectedGauss-
erosion depthandnormalmapsrendering.WetraintheGaussianrepresen-
ianpivots,nottoalltetrahedralvertices,preventingcollapse.This
tationusingstandard3DGaussianSplattingtechniqueswithaggres-
ensuresregularizationiseffectivewithoutharmingmeshintegrity.
sivedensificationandimportance-weightedsamplinginspiredby
Interiorartifacts. Thepreviouslossesdonotprovideadequate Mini-Splatting2[FangandWang2024b].Weprovidemoredetails
constraintsonoccludedpartsofthescene.Asaconsequence,the ontheoptimizationprocedure,SDFnormalization,meshrendering
interiorofthesurfacemesh,whichdoesnotreceivepropersupervi- andDelaunaytriangulationinthesupplementarymaterial.
sionthroughdepthandnormalrenderings,tendstoproducechaotic
Datasets. Weevaluateourmethodonseveralstandarddatasets:
structuresandinternalcavities,ratherthanbeingemptyasitshould
(1)theTanksandTemples(T&T)dataset[Knapitschetal.2017],
be.Theseinteriorartifactscanimpactdownstreamapplicationssuch
whichcontainscomplexreal-worldsceneswithvaryingscalesand
asmesheditingorphysicssimulations.
challenging geometry; (2) the DTU dataset [Jensen et al. 2014],
Topreventsuchartifactsinsidethemesh,weintroduceanovel
whichprovidesscansofrealobjectswithgroundtruthgeometry;
loss based on a feedback loop between the SDF values and the
(3)theMip-NeRF360dataset[Barronetal.2022],whichcontains
extractedmesh.Thislossaimstoenforceoccludedpointslocated
unboundedandcomplexscenes;(4)andDeepBlending[Hedman
inside the mesh to have negative SDF values. Specifically, after
etal.2018]whichcontainsreal-worldsceneswithchallengingge-
extractingthesurfaceusingtheGaussiansandtheirSDFvaluesas
ometry.ForT&T,wefollowpreviousworks[Huangetal.2024;Yu
describedinsection4.3,weusethemeshtobuildanoccupancy
etal.2024;Zhangetal.2024]andusethescenesBarn,Caterpillar,
label?
?
?{0,1}foreachDelaunaysite?,indicatingwhetherthese
Ignatius,Courthouse,Meetingroom,andTruck.ForDTU,weuse
pointsareinsideoroutsidethevisibleportionofthesurfacemeshas
thesame15scenesasinpreviousworks[Huangetal.2024;Yuetal.
describedbelow.WefinallyenforcetheDelaunayverticeslabeledas
2024;Zhangetal.2024].
insidetohavenegativeSDFvalues,withoutenforcinganyadditional
constraintforDelaunayverticeslabeledasoutsideùasthesepoints Metrics. For quantitative evaluation of surface reconstruction
alreadyreceivesupervisionfromtherenderings. quality,wefollowstandardpractice[Yuetal.2024;Zhangetal.
Thisregularizationtermiscomputedasfollows: 2024]andreportF1-scoreforT&TandChamferDistanceforDTU.
FornovelviewsynthesisonMip-NeRF360andDeepBlending,we
??
L interior = ?(?(?? ?),? ?)╖? ? , (9) reportPSNR,SSIM,andLPIPSmetrics.
? Nodatasetprovidesgroundtruthgeometryforbackgroundob-
where? isthecross-entropyloss,?isthesigmoidfunction,? ? is jects.TomitigatethisissueweuseMesh-BasedNovelViewSyn-
theSDFvalueoftheDelaunaysite?,and? ? istheoccupancylabel thesis (seeSection.6.3)tomeasurethevisualconsistencybetween
of?. theextractedmeshandreferenceviews,whichcontainbackground
Tocomputetheoccupancylabels? ?,wefirstrenderdepthmapsof information.Thisevaluationmeasurestheabilityofreconstructed
themeshfromalltrainingviewpoints.Then,foranyDelaunaysite?, meshestoaccuratelyrepresentfullscenes,includingbackground
weclassifyitasinsidethemeshifitlocatedôbehindöalldepthmaps objects,whilerequiringonlyasetofgroundtruthimages.
containing?intheirfieldofview.Updatingtheoccupancylabels
6.1 ResourcesRequirements.
requiresafewsecondsandcanbeperformedevery200iterations,
resultinginacomputationallyefficientandeffectiveregularization Theadditionofmesh-basedconstraintsinevitablyleadstoincreased
termforproducingmesheswithcleaninteriors. computationalload.However,weobservedthatonlyasubsetof
Ourfulloptimizationlossisfinallydefinedas: Gaussiansiscriticalforproducinganeffectivetetrahedralstructure.
Buildingonthis,inspiredbythesamplingstrategyintroducedin
L=L +L +L , (10) Mini-Splatting2[FangandWang2024b],weemployitforadistinct
vol mesh reg
purpose:selectingtheGaussianswhichwillspawntheGaussian
with
pivotsthatserveasDelaunayvertices.UnlikeMini-Splatting2,our
densemodelpreservesallGaussiansthroughoutoptimization,and
L =? L +? L , (11)
reg erosion erosion interior interior thesamplingcriterionisusedexclusivelytoidentifythosemost
where? and? arehyperparameterscontrollingthe suitableformeshconstructionratherthantoprunetherepresenta-
erosion interior
strengthoftheanti-erosionandinteriorregularization,respectively. tion.
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

1:8 ò AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,GeorgeDrettakis,andMaksOvsjanikov
GroundTruthImages
(a)2DGS
(b)GOF
(c)RaDe-GS
(d)ours
Fig.5. QualitativecomparisonofsurfacereconstructionresultsontheTanksandTemplesdataset.MILoproducesmeshesoffullscenesthatbetter
fitthetargetsurfaces,withreducedartifactssuchascavities(threeleftimages).Ourapproacheffectivelyaddressestheerosionproblemthatplaguesprevious
methods(tworightimages),whileproducinglightermeshes.Inaddition,itisparticularlyeffectiveatrecoveringperipheralobjects(e.g.,thebikerackor
chandelier)thatareoftenmissedbypost-hocmethods.
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

MILo:Mesh-In-the-LoopGaussianSplattingforDetailedandEfficientSurfaceReconstruction ò 1:9
Table3. SurfacereconstructionmetricsontheDTUdataset.WereporttheChamferDistanceacross15scenes.Ourmethodachievescompetitive
performancewithsignificantlyfewerGaussiansandmeshverticescomparedtopreviousapproaches.
24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 Mean Time
ticilpmi
NeRF[Mildenhalletal.2020] 1.90 1.60 1.85 0.58 2.28 1.27 1.47 1.67 2.05 1.07 0.88 2.53 1.06 1.15 0.96 1.49 >12h
VolSDF[Yarivetal.2021] 1.14 1.26 0.81 0.49 1.25 0.70 0.72 1.29 1.18 0.70 0.66 1.08 0.42 0.61 0.55 0.86 >12h
NeuS[Wangetal.2021] 1.00 1.37 0.93 0.43 1.10 0.65 0.57 1.48 1.09 0.83 0.52 1.20 0.35 0.49 0.54 0.84 >12h
Neuralangelo[Lietal.2023] 0.37 0.72 0.35 0.35 0.87 0.54 0.53 1.29 0.97 0.73 0.47 0.74 0.32 0.41 0.43 0.61 >12h
ticilpxe
3DGS[Kerbletal.2023] 2.14 1.53 2.08 1.68 3.49 2.21 1.43 2.07 2.22 1.75 1.79 2.55 1.53 1.52 1.50 1.96 5.2m
SuGaR[GuΘdonandLepetit2024b] 1.47 1.33 1.13 0.61 2.25 1.71 1.15 1.63 1.62 1.07 0.79 2.45 0.98 0.88 0.79 1.33 52m
2DGS[Huangetal.2024] 0.48 0.91 0.39 0.39 1.01 0.83 0.81 1.36 1.27 0.76 0.70 1.40 0.40 0.76 0.52 0.80 8.9m
GOF[Yuetal.2024] 0.50 0.82 0.37 0.37 1.12 0.74 0.73 1.18 1.29 0.68 0.77 0.90 0.42 0.66 0.49 0.74 55m
RaDe-GS[Zhangetal.2024] 0.46 0.73 0.33 0.38 0.79 0.75 0.76 1.19 1.22 0.62 0.70 0.78 0.36 0.68 0.47 0.68 8.3m
Ours(base) 0.43 0.74 0.34 0.37 0.80 0.74 0.70 1.21 1.22 0.66 0.62 0.80 0.37 0.76 0.48 0.68 25m
Thistargetedselectionyieldslighterandmoreaccuratemeshes,
significantlyreducingthecomputationalcostoftriangulation,March-
ingTetrahedra,andmeshrasterization.Table1providesacompre-
hensivecomparisonofthecomputationalresourcesrequiredby
concurrentmethods.
MemoryRequirements. Allexperimentswereconductedonasin-
gleNVIDIARTX4090GPUwith24GBofVRAM.Thebasemodel
(a)Ours (b)GOF
containsbetween0.1and0.5millionGaussians,requiring10GBof
VRAMduringtraining.Thedensemodelgenerallycontainsbetween 60
2and4millionGaussiansrequiringupto17GB.
TimeRequirements. Acompletetrainingrunforthebasemodel 40
onasingleGPUtakes25minutesforboundedscenesfromDTU
andbetween40and50minutesforunboundedscenesfromTanks
20
andTemplesandMip-NeRF360,whilethedensemodelrequiresup
to2hoursforunboundedscenes.
0
StorageRequirements. Ourbasemodelusessignificantlyfewer 0.0 0.5 1.0
Normalized Distance
Gaussians(0.1-0.5M)andproducesmesheswithfewerverticescom-
paredtopreviousapproaches.Despiteusingfewerresources,our
methodachievessuperiorperformanceintermsofsurfacerecon-
structionquality.Thedensevariantofourmodelmaintainsalarger
setofGaussiansbutstillproducesmesheswithfewerverticesand
trianglesthancompetingmethods,makingitmoresuitablefordown-
streamapplicationsthatrequireefficientmeshrepresentations.
6.2 SurfaceReconstruction
Table2presentsthequantitativeresultsontheTanksandTemples
dataset.Ourmethodconsistentlyoutperformspreviousapproaches
intermsofF-score,demonstratingtheeffectivenessofourmesh-
in-the-loopoptimizationstrategy.Thedensevariantofourmethod
achievesthebestresultsamongallGaussian-basedapproaches.
WeprovideresultsontheDTUdataset,consistingofsmall,object-
centricscenes.Table3showstheresultsonthisdataset.Notethat,
DTUconsistsofisolatedobjectsinhighlycontrolledscenes,where
standardpost-hocmeshextractionisalreadyeffective.MILoispar-
ticularlyeffectiveforcomplexfull-scenereconstruction.Still,our
mesh-in-the-loopoptimizationmaintainscompetitiveperformance
intermsofChamferDistance(CD)onDTU.Inlinewithestablished
practice,weevaluateperformanceusingF-ScoreontheTanksand
TemplesdatasetandChamferDistance(CD)onDTU.
Figure5showsqualitativecomparisonsofourmethodwithpre-
viousapproachesonselectedscenesfromtheTanksandTemples
)%(
stnioP
fo
rebmuN
Ours (Base)
Ours (Dense)
GOF
RaDe-GS
(c)AveragecumulativedistancedistributionontheT&Tdataset
Fig.6. Reconstruction?GroundtruthDistanceEvaluation.Qualita-
tively,ourreconstructionsusingthedifferentiablerasterizerfromRaDe-GS
(a)areclosertothegroundtruthpointcloudsthanbothGOF(b)and
RaDe-GS.TheT&T[Knapitschetal.2017]evaluationtoolkitreportscu-
mulativehistograms(c)showingtheproportionofreconstructedpoints
withinagivendistancefromthegroundtruthsurface.Thedashedline
marksthethresholdusedtocomputetheF1-score.Ourmethodproduces
significantlymorepointsclosetothetruesurface(withinanormalized
distanceof1.0),demonstratinghigherreconstructionaccuracycompared
topriorapproaches.
andMip-NeRF360datasets.Ourmethodproducesmesheswith
significantlycleanersurfacesandbetterpreservationoffinede-
tails.Inparticular,ourapproacheffectivelyaddressestheerosion
problempresentinpreviousmethods,resultinginmorecomplete
reconstructionsofthinstructuresandcomplexgeometry:seeFig.6
foradetailedanalysis.
6.3 Mesh-BasedNovelViewSynthesis
Despiteextensiveresearchonmeshreconstructionfromimages,cur-
rentevaluationbenchmarksremainlimited.CommondatasetsùDTU,
MipNeRF360,andTanks&Templesùallhavenotabledrawbacks:
MipNeRF360lacksground-truthgeometry;DTUfeaturesoverly
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

1:10 ò AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,GeorgeDrettakis,andMaksOvsjanikov
Table4. Mesh-BasedNovelViewsynthesismetricsonreal-worlddatasets.WereportPSNR,SSIM,andLPIPSmetrics,aswellasthetotalnumber
ofGaussiansandvertices(inmillions).Ourmethoddemonstratessuperiorperformanceonthesechallengingunboundedscenes.Ours(base)usesthe
differentiablerasterizerfromRaDe-GS.
MipNeRF360 Tanks&Temples DeepBlending
PSNR? SSIM? LPIPS? #Gaussians? #Verts? PSNR? SSIM? LPIPS? #Gaussians? #Verts? PSNR? SSIM? LPIPS? #Gaussians? #Verts?
2DGS 15.36 0.4987 0.4749 1.88 4.31 14.23 0.5697 0.4854 0.98 16.39 17.46 0.7133 0.5556 1.51 16.21
GOF 20.78 0.5730 0.4654 2.99 32.80 21.69 0.6904 0.3261 1.25 11.63 27.44 0.8078 0.2549 1.64 12.64
RaDe-GS 23.56 0.6685 0.3612 2.91 30.85 20.51 0.6595 0.3447 1.18 10.06 26.74 0.7964 0.2621 1.65 11.53
Ours(base) 24.09 0.6885 0.3235 0.46 6.73 21.46 0.7067 0.3489 0.28 4.36 28.04 0.8336 0.2285 0.38 5.40
controlled,object-centricscenes;andTanks&Templesprovides testingviews,asachievinggoodmesh-basedrenderingperformance
sparseground-truthlimitedtoforegroundregions.Moreover,evalu- onasparsetestsetdoesnotnecessarilytranslateintohighquality
ationprotocolstomeasurethealignmentofextractedsurfaceswith oftheunderlyinggeometry.
groundtruthimagesremainuncommoninsurfacereconstruction Table4showshowMesh-BasedNovelViewSynthesisallowsus
research. tocompareagainstotherstate-of-the-artmethods.Weusethediffer-
Wehighlightthisasanimportantissueinsurfacereconstruction. entiablerasterizerfromRaDe-GSfortrainingOurs(base).Ourap-
Inspiredbypreviousworkswhichevaluatesurface-basedviewsyn- proachoutperforms2DGS[Huangetal.2024]andRaDe-GS[Zhang
thesismethods[Reiseretal.2024],weuseMesh-BasedNovelView etal.2024]acrossallmetrics.GOF[Yuetal.2024]produceshighlyde-
Synthesisasanevaluationmethodformeshesthatreliessolelyon tailedmeshes,yieldingstrongPSNRresults,butourmethodachieves
ground-truthRGBimages.Thecoreintuitionisthatbettergeometry comparablePSNRwhileoutperforminginSSIMandLPIPS.This
shouldyieldbettermesh-basedrenderings:foreachtestview,we suggests that our meshes are not only accurate but also exhibit
renderthesceneusingthemeshandcomparetheresultingimage fewervisualartifactsandlessnoisecomparedtothosefromGOF.
tothecorrespondingground-truthimage. Weincludeaqualitativecomparisoninthesupplementarymaterial.
Bymeasuringthevisualconsistencybetweenmeshrenderings Moreover,theperformanceonTanks&TemplesundertheMesh-
andreferenceviews,thismetricallowsustoevaluate: BasedNovelViewSynthesis metric(Table4)showsstrongcorre-
lationwiththeF1scorecomputedagainstground-truthgeometry
ò Geometric artifacts and misalignment between the recon-
(Table2).Forexample,amongthecommonbaselines,2DGSand
structedsurfaceandthegroundtruthimages(e.g.,surface
RaDe-GSrankfourthandthird,respectively,underbothmetrics.
erosionorinflation);
Ours(base)andGOFalsoperformconsistently,withtheOurs(base)
ò Meshcompleteness,sincemissinggeometryusuallyresults
achievingaslightlyhigherF1scoreandGOFperformingmarginally
indegradedrenderingperformance;
betteronMesh-BasedNovelViewSynthesis.Theseresultsindi-
ò Backgroundreconstruction,evenwhen3Dgroundtruthis
catethatourproposedmetriciswellalignedwithground-truth
unavailable.
geometricaccuracy.
Torenderthemeshfromagivenviewpoint,wefirstrasterizethe
6.4 NovelViewSynthesis
trianglesusingNvdiffrast[Laineetal.2020].Then,weassociatea
colorwitheachpixelbasedontherasterizedtriangle. Whileourprimaryfocusisonsurfacereconstruction,wealsopro-
Anaiveapproachwouldrelyonvertexcolors,assigningRGB videinsupplementarymaterialanevaluationofthenovelviewsyn-
valuestoeachmeshvertexandinterpolatingthosevaluestodeter- thesisqualityofouroptimizedGaussians.Ourmethodmaintains
mineapixelÆscolor.However,thismethodbiasesthemetrictowards competitiverenderingqualitycomparedtopreviousapproaches,
densemeshes,asimagequalitywouldsufferforsparse,yetaccurate, demonstrating that our mesh-in-the-loop optimization does not
meshesduetolimitedcolorresolution. compromisethevisualfidelityoftheGaussianrepresentation.
Toovercomethis,wedecouplecolorfrommeshresolutionby
employinganeuralcolorfield? : R3 ? [0,1]3 fortexturing
color
themesh.Specifically,toobtaintherenderedcolorforanypixel
?,wefirstcalculatethe3Dlocation? ? R3 ofthetrianglepoint
rasterizedontothepixel?.Thisisachievedbybackprojectingthe
depthvalueusingcameraparameters.Wethenquerytheneural
colorfieldatthisbackprojectedsurfacepointtoretrieveitsRGB
value? (?).Unlikevertexcolors,thismeshtexturingprocess
storesc
c
o
o
l
l
o
or
rvaluesinaneuralfieldratherthanper-vertex,ensuring
(a)Baseline+L
MD
(b)Baseline+L
mesh
colorassignmentisindependentofmeshresolution.
Fig.7. EffectofaddingNormalsupervision.Despiteshowcasingan
Inpractice,foreachmeshunderevaluation,wefirsttrainthe
apparentdecreaseinperformanceinourablationstudy(seethesupplemen-
neuralfieldusingonlythetrainingviews.WeadoptTensoRF[Chen tarymaterial),thisfiguresupportsourstatementthatthecombinationof
etal.2022]asthebackbonerepresentationandoptimizeitfor5k depthandnormalsupervisionisessential.Theresultingmeshwhenusing
iterations.Wethenrendertestviewsforevaluation.Thismetric onlyLMD issignificantlynoisierthanwhensupervisingwithLmesh .
effectivelyquantifiesthealignmentbetweenmeshgeometryand
theimagedata.Notethat,thisevaluationprotocolassumesdense
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

MILo:Mesh-In-the-LoopGaussianSplattingforDetailedandEfficientSurfaceReconstruction ò 1:11
Table5. AblationstudyontheTanksandTemplesdataset.Wereport
drivesupmemoryrequirementsasresolutionincreases,andfrom
onlytheaverage? -scoreacrossallscenestoemphasizetheimpactofeach
1 itstendencytooversmoothgeometricdetail.
component.
InFigure9,wepresentaquantitativecomparisonbetweenour
extractionmethodandTSDFfusion.ForTSDFfusion,weoptimized
Method Average? 1-score?
Gaussians with the same densification strategy as MILo, but re-
Baseline 0.41
movedthemesh-in-the-loopregularizationtermandemployedthe
+L MD 0.46
samelossfunctionasin[Huangetal.2024;Yuetal.2024;Zhangetal.
+L mesh 0.44
+L mesh+L erosion 0.44 2024]toavoidbias.TSDFfusionrequiressubstantialmemoryand
Ours(Baseline+L mesh +L reg) 0.47 ispronetoout-of-memoryfailures;consequently,weexcludethe
Ours(GOF) 0.49 Courthousescenefromthisbenchmark,asTSDFfusioncouldnot
berunacrossallmeshresolutionsduetomemoryconstraints.As
showninFigure9,ourmethodconsistentlyachieveshigherrecon-
structionquality,demonstratingtherobustnessofourlearnableSDF
strategy.Wehighlightthat,TSDFfusionrequiresiteratingover
alltrainingviewpointsforeachextraction,leadingtoalong
andcomputationallyexpensiveprocessthatpreventsintegration
intothetrainingloopofGaussianSplattingrepresentations.
Bycontrast,MILoleverageslearnableSDFvaluesandascalable
(a)Withoutinteriorregularization (b)Withinteriorregularization pivotsettoenablelightweightandefficientmeshextraction,making
itpracticaltoincorporatemeshextractiondirectlyintothetraining
Fig.8. Comparisonofmeshinteriors.Weshowacross-sectionofthe
loop.
reconstructedmeshoftheBarnscenetorevealitsinternalstructure.Our
interiorregularizationLinterior effectivelyeliminatesinternalartifacts,pro-
ducingclean,watertightmesheswithemptyinteriors,whichiscrucialfor
downstreamapplicationssuchasphysicssimulationsandanimation.
50
45
6.5 AblationStudy
40
Toevaluatethecontributionofeachcomponent,weconductabla-
tionstudiesontheTanksandTemplesdataset(seeTab.5).Adding 35
depthsupervisionviathemeshloss(+L )substantiallyboosts
MD
? -score,whilefurtherincludingthenormalloss(+L )slightly 30
1 mesh
reduces it. Still, Fig. 7 shows that combining both is critical for
removingnoisefrommeshes. 25
Introducingtheanti-erosionloss(+L +L )improvesre- 2 4 6 8 10 12 14 16
mesh erosion
Number of Vertices (╫1e6)
constructionbypreservingthinstructuresandfinedetailsùespecially
inchallengingregionslikefencesorvegetationùbypenalizingsur-
faceerosion.Ourlossalsoactsasamechanismtorecovergeometry
lostduetoSDFsignflipsyieldingamodest? -scoregain(Fig.5). 1
Fig.8highlightsthebenefitofourinteriorregularization,which
eliminatesinternalcavitiesandartifacts,producingwatertightsur-
facessuitablefordownstreamtaskslikephysicssimulation.
Combiningallcomponentsleadstothehighest? -score,validat-
1
ingtheeffectivenessofourfullmethod.Wealsodemonstrateits
plug-and-playnaturebyintegratingitintoGOFÆspipeline.
6.6 AnalysisofMeshExtraction
AsdiscussedinSection2,manyexistingapproachestomeshex-
tractionfromimagesrelyontheTSDFalgorithm.Thesemethods
typicallyfollowasimpleyetseeminglyeffectivepipeline:theyfirst
estimateaccuratedepthmapsusingavarietyoftechniques,and
thenfusethesemapsintoameshviaTSDF.However,weargue
thatTSDFfusiondoesnotscalewelltolarge-scalereconstructions.
Thislimitationarisesfromitsrelianceonafixed3Dgrid,which
erocS-1F
MILo extraction (Ours)
TSDF Fusion
Fig.9. QuantitativeevaluationofmeshextractionmethodsontheTanks
&Templesdatasetacrossdifferentmeshresolutions.Wecompare(a)tra-
ditionalTSDFfusion,whichreliesonaregular3Dgrid,with(b)ourmesh
extractionmethod,whichlearnsSDFvaluesatGaussianpivots.
6.7 LimitationsandConclusions
Whileourmethodsignificantlyimprovessurfacereconstruction
quality,itstillhassomelimitations.First,thecomputationalcostof
extractingandprocessingthemeshateveryiterationincreasesthe
trainingtimecomparedtostandardGaussianSplatting,althoughit
stillremainsmanageablecomparedtobaselines.Second,thequality
of the reconstruction still depends on the initial distribution of
Gaussians,whichmaynotbeoptimalforallscenes.
Conceptually,ourframeworkunlocksmanysurface-basedpro-
cessingtoolsforGaussiansthatcanbeappliedduringtraining.This
integrationofmesh-basedoperationswithGaussianrepresentations
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

1:12 ò AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,GeorgeDrettakis,andMaksOvsjanikov
opensnewpossibilitiesforgeometrymanipulation,regularization, applications(SeattleWashingtonUSA).ACM,127û134. doi:10.1145/781606.781627
andenhancementthroughouttheoptimizationprocess,notjustas AkioDoiandAkioKoide.1991. Anefficientmethodoftriangulatingequi-valued
surfacesbyusingtetrahedralcells. IEICETRANSACTIONSonInformationand
apost-processingstep.
Systems74,1(1991),214û224.
Futureworkcouldexplorenoveladaptivesamplingstrategies TimElsner,VictorCzech,JuliaBerger,ZainSelman,IsaakLim,andLeifKobbelt.2023.
forDelaunaysites,andimprovedinitializationtechniques.Addi- AdaptiveVoronoiNeRFs.arXiv:2303.16001[cs] http://arxiv.org/abs/2303.16001
GuangchiFangandBingWang.2024a. Mini-splatting:Representingsceneswith
tionally,extendingourapproachtodynamicscenesandexploring aconstrainednumberofgaussians.InEuropeanConferenceonComputerVision.
itsapplicationsinreal-timerenderingandinteractiveapplications Springer,165û181.
GuangchiFangandBingWang.2024b.Mini-Splatting2:Building360Sceneswithin
wouldbepromisingdirectionsforfutureresearch.
MinutesviaAggressiveGaussianDensification. arXivpreprintarXiv:2411.12788
Furtherexplorationofsurface-basedprocessingtoolsthatcan (2024).
beintegratedintothetrainingpipelinecouldalsoyieldsignificant SaraFridovich-Keil,AlexYu,MatthewTancik,QinhongChen,BenjaminRecht,and
AngjooKanazawa.2022.Plenoxels:RadianceFieldswithoutNeuralNetworks.In
improvementsinreconstructionqualityandversatility.
CVPR.5501û5510.
Finally,throughaamesh-basednovelviewsynthesisevaluation DiegoGomez,BingchenGong,andMaksOvsjanikov.2025.FourieRF:Few-ShotNeRFs
wehighlightMthelackofstandardizedquantitativeprotocolsto viaProgressiveFourierFrequencyControl.In2025InternationalConferenceon3D
Vision(3DV).607û615.doi:10.1109/3DV66043.2025.00061
evaluatethealignmentofthereconstructedsurfacewiththetraining ShrisudhanGovindarajan,DanielRebain,KwangMooYi,andAndreaTagliasacchi.
images.ThisistrueonwidelyuseddatasetssuchasDTU,MipNeRF 2025.RadiantFoam:Real-TimeDifferentiableRayTracing.arXiv:2502.01157[cs]
doi:10.48550/arXiv.2502.01157
360,andTanks&Temples.Whilethisapproachoffersasteptoward
AntoineGuΘdon,TomokiIchikawa,KoheiYamashita,andKoNishino.2025.MAtCha
addressingthisgap,itremainsaninitialeffort;futureworkcould Gaussians:AtlasofChartsforHigh-QualityGeometryandPhotorealismFrom
focus on refining the evaluation methodology to better capture SparseViews.InCVPR.
AntoineGuΘdonandVincentLepetit.2024a. GaussianFrosting:EditableComplex
reconstructionqualityinchallengingreal-worldscenes.
RadianceFieldswithReal-TimeRendering.InECCV.
AntoineGuΘdonandVincentLepetit.2024b.SuGaR:Surface-AlignedGaussianSplatting
7 Acknowledgements forEfficient3DMeshReconstructionandHigh-QualityMeshRendering.InCVPR.
5354û5363.
PartsofthisworkweresupportedbytheERCConsolidatorGrant PeterHedman,JulienPhilip,TruePrice,Jan-MichaelFrahm,GeorgeDrettakis,and
GabrielJ.Brostow.2018.Deepblendingforfree-viewpointimage-basedrendering.
ôVEGAö(No.101087347),theERCAdvancedGrantôexplorerö(No.
ACMTOG37,6(2018),257.doi:10.1145/3272127.3275084
101097259),theANRAIChairAIGRETTE.Co-fundedbytheEuro- BinbinHuang,ZehaoYu,AnpeiChen,AndreasGeiger,andShenghuaGao.2024.2D
peanUnion(EU)(ERCAdvancedgrantFUNGRAPHNo788065and GaussianSplattingforGeometricallyAccurateRadianceFields.InSIGGRAPH2024
ConferencePapers.AssociationforComputingMachinery. doi:10.1145/3641519.
ERCAdvancedGrantNERPHYSNo101141721).Viewsandopinions 3657428
expressedarehoweverthoseoftheauthor(s)onlyanddonotnec- BinbinHuang,ZehaoYu,AnpeiChen,AndreasGeiger,andShenghuaGao.2024.2D
GaussianSplattingforGeometricallyAccurateRadianceFields.InACMSIGGRAPH
essarilyreflectthoseoftheEUortheEuropeanResearchCouncil.
ConferencePapers.1û11.
NeithertheEUnorthegrantingauthoritycanbeheldresponsible RasmusJensen,AndersDahl,GeorgeVogiatzis,EngilTola,andHenrikAanµs.2014.
forthem. Largescalemulti-viewstereopsisevaluation.In2014IEEEConferenceonComputer
VisionandPatternRecognition.IEEE,406û413.
BernhardKerbl,GeorgiosKopanas,ThomasLeimkⁿhler,andGeorgeDrettakis.2023.
References 3DGaussianSplattingforReal-TimeRadianceFieldRendering. ACMTOG42,4
(July2023). https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
NinaAmenta,MarshallBern,andManolisKamvysselis.1998.AnewVoronoi-based ShakibaKheradmand,DanielRebain,GopalSharma,WeiweiSun,Yang-CheTseng,
surfacereconstructionalgorithm.InProceedingsofthe25thannualconferenceon HossamIsack,AbhishekKar,AndreaTagliasacchi,andKwangMooYi.2024. 3d
Computergraphicsandinteractivetechniques-SIGGRAPHÆ98.ACMPress,415û421. gaussiansplattingasmarkovchainmontecarlo.AdvancesinNeuralInformation
doi:10.1145/280814.280947 ProcessingSystems37(2024),80965û80986.
NinaAmenta,SungheeChoi,andRaviKrishnaKolluri.2001. Thepowercrust.In ArnoKnapitsch,JaesikPark,Qian-YiZhou,andVladlenKoltun.2017. Tanksand
ProceedingsofthesixthACMsymposiumonSolidmodelingandapplications(New temples:Benchmarkinglarge-scalescenereconstruction. ACMTransactionson
York,NY,USA)(SMAÆ01).AssociationforComputingMachinery,249û266. doi:10. Graphics(ToG)36,4(2017),1û13.
1145/376957.376986 ArnoKnapitsch,JaesikPark,Qian-YiZhou,andVladlenKoltun.2017. Tanksand
FranzAurenhammer.1991.Voronoidiagramsùasurveyofafundamentalgeometric Temples:BenchmarkingLarge-ScaleSceneReconstruction.ACMTOG36,4(2017).
datastructure.ACMcomputingsurveys(CSUR)23,3(1991),345û405. SamuliLaine,JanneHellsten,TeroKarras,YeonghoSeol,JaakkoLehtinen,andTimo
JonathanTBarron,BenMildenhall,MatthewTancik,PeterHedman,RicardoMartin Aila.2020. ModularPrimitivesforHigh-PerformanceDifferentiableRendering.
Brualla,andPratulPSrinivasan.2021.Mip-NeRF:AMultiscaleRepresentationfor ACMTOG39,6(2020).
Anti-AliasingNeuralRadianceFields.InICCV.5855û5864. ShujuanLi,Yu-ShenLiu,andZhizhongHan.2025.Gaussianudf:Inferringunsigned
JonathanT.Barron,BenMildenhall,DorVerbin,PratulP.Srinivasan,andPeterHedman. distancefunctionsthrough3dgaussiansplatting.InProceedingsoftheComputer
2022. Mip-NeRF360:UnboundedAnti-AliasedNeuralRadianceFields.InCVPR. VisionandPatternRecognitionConference.27113û27123.
IEEE,5460û5469.doi:10.1109/CVPR52688.2022.00539 ZhaoshuoLi,ThomasMⁿller,AlexEvans,RussellHTaylor,MathiasUnberath,Ming-
JonathanT.Barron,BenMildenhall,DorVerbin,PratulP.Srinivasan,andPeterHedman. YuLiu,andChen-HsuanLin.2023. Neuralangelo:High-fidelityneuralsurface
2023.Zip-NeRF:Anti-AliasedGrid-BasedNeuralRadianceFields.InICCV.19697û reconstruction.InProceedingsoftheIEEE/CVFConferenceonComputerVisionand
19705. PatternRecognition.8456û8465.
AnpeiChen,ZexiangXu,AndreasGeiger,JingyiYu,andHaoSu.2022. TensoRF: RongLiu,DylanSun,MeidaChen,YueWang,andAndrewFeng.2025.Deformable
TensorialRadianceFields.InECCV.333û350. BetaSplatting.arXivpreprintarXiv:2501.18630(2025).
HanlinChen,ChenLi,andGimHeeLee.2023. NeuSG:NeuralImplicitSurfaceRe- SaswatSubhajyotiMallick,RahulGoel,BernhardKerbl,MarkusSteinberger,Fran-
constructionwith3DGaussianSplattingGuidance.arXivpreprintarXiv:2312.00846 ciscoVicenteCarrasco,andFernandoDeLaTorre.2024.Taming3dgs:High-quality
(2023). radiancefieldswithlimitedresources.InSIGGRAPHAsia2024ConferencePapers.
HanlinChen,FangyinWei,ChenLi,TianxinHuang,YunsongWang,andGimHeeLee. 1û11.
2024.VCR-GauS:ViewConsistentDepth-NormalRegularizerforGaussianSurface NissimMaruani,RomanKlokov,MaksOvsjanikov,PierreAlliez,andMathieuDesbrun.
Reconstruction.InNeurIPS. 2023.VoroMesh:LearningWatertightSurfaceMesheswithVoronoiDiagrams.In
PinxuanDai,JiaminXu,WenxiangXie,XinguoLiu,HuaminWang,andWeiweiXu. 2023IEEE/CVFInternationalConferenceonComputerVision(ICCV).14565û14574.
2024.High-qualitysurfacereconstructionusinggaussiansurfels.InACMSIGGRAPH https://openaccess.thecvf.com/content/ICCV2023/html/Maruani_VoroMesh_
2024conferencepapers.1û11. Learning_Watertight_Surface_Meshes_with_Voronoi_Diagrams_ICCV_2023_
TamalK.DeyandSamratGoswami.2003. Tightcocone:awater-tightsurfacere- paper.html
constructor.InProceedingsoftheeighthACMsymposiumonSolidmodelingand
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

MILo:Mesh-In-the-LoopGaussianSplattingforDetailedandEfficientSurfaceReconstruction ò 1:13
SupplementaryMaterial
NissimMaruani,MaksOvsjanikov,PierreAlliez,andMathieuDesbrun.2024.PoNQ:A
NeuralQEM-BasedMeshRepresentation.In2024IEEE/CVFConferenceonComputer A Preliminaries
VisionandPatternRecognition(CVPR)(Seattle,WA,USA).IEEE,3647û3657. doi:10.
1109/CVPR52733.2024.00350 Inthissupplementarymaterial,westartbybrieflyreviewingthe
BenMildenhall,PratulP.Srinivasan,MatthewTancik,JonathanT.Barron,RaviRa-
mamoorthi,andRenNg.2020.NeRF:RepresentingScenesasNeuralRadianceFields keyconceptsthatformthefoundationofourapproach:3DGaussian
forViewSynthesis.InECCV.99û106. SplattingandDelaunaytetrahedralization.
NicolasMoenne-Loccoz,AshkanMirzaei,OrPerel,RiccardodeLutio,JanickMar-
tinezEsturo,GavrielState,SanjaFidler,NicholasSharp,andZanGojcic.2024.3D
GaussianRayTracing:FastTracingofParticleScenes.ACMTransactionsonGraphics A.1 3DGaussianSplatting
(TOG)43,6(2024),1û19.
ThomasMⁿller,AlexEvans,ChristophSchied,andAlexanderKeller.2022. Instant 3DGaussianSplatting(3DGS)[Kerbletal.2023]hasemergedasa
NeuralGraphicsPrimitiveswithaMultiresolutionHashEncoding.ACMTOG41,4, powerfultechniquefornovelviewsynthesis,offeringhigh-quality
Article102(July2022),15pages.doi:10.1145/3528223.3530127
renderingwithreal-timeperformance.Inthisrepresentation,a3D
J.Munkberg,J.Hasselgren,T.Shen,J.Gao,W.Chen,A.Evans,T.Mⁿller,andS.Fidler.
2022.ExtractingTriangular3DModels,Materials,andLightingFromImages.In sceneismodeledasacollectionof3DGaussians,eachdefinedby
CVPR. itsposition? ?R3,covariancematrix??R3╫3(typicallyparame-
ChristianReiser,StephanGarbin,PratulP.Srinivasan,DorVerbin,RichardSzeliski,Ben terizedbyascalingvector? ? R3andarotationmatrix? ? R3╫3
Mildenhall,JonathanT.Barron,PeterHedman,andAndreasGeiger.2024.Binary
OpacityGrids:CapturingFineGeometricDetailforMesh-BasedViewSynthesis. encodedusingaquaternion? ?R4),opacity? ? (0,1),andappear-
SIGGRAPH(2024). anceattributes(suchassphericalharmonicscoefficients? ?R? for
T.Shen,J.Munkberg,J.Hasselgren,K.Yin,Z.Wang,W.Chen,Z.Gojcic,S.Fidler,N.
Sharp,andJ.Gao.2023. FlexibleIsosurfaceExtractionforGradient-BasedMesh view-dependentcolor).Duringrendering,these3DGaussiansare
Optimization.ACMTOG42,4(2023). projectedontotheimageplaneas2DGaussiansandcompositedin
DorVerbin,PeterHedman,BenMildenhall,ToddZickler,JonathanT.Barron,and
afront-to-backordertoproducethefinalimage.
PratulP.Srinivasan.2022.Ref-NeRF:StructuredView-DependentAppearancefor
NeuralRadianceFields.CVPR(2022).
PengWang,LingjieLiu,YuanLiu,ChristianTheobalt,TakuKomura,andWenping A.2 DelaunayTriangulation
Wang.2021.NeuS:Learningneuralimplicitsurfacesbyvolumerenderingformulti-
viewreconstruction.InProceedingsofthe35thInternationalConferenceonNeural Givenasetof3Dpoints,theDelaunaytriangulationdividestheir
InformationProcessingSystems(NIPS).Article2081,13pages.
convexhullintotetrahedrasuchthatnopointliesinsidethecir-
YimingWang,QinHan,MarcHabermann,KostasDaniilidis,ChristianTheobalt,and
LingjieLiu.2023.Neus2:Fastlearningofneuralimplicitsurfacesformulti-view cumsphereofanytetrahedron.Thisstructurehasbeenextensively
reconstruction.InProceedingsoftheIEEE/CVFInternationalConferenceonComputer studied due to its desirable theoretical and practical properties;
Vision.3295û3306.
see[Aurenhammer1991]formoredetails.
FrederikWarburg,EthanWeber,MatthewTancik,AleksanderHolynski,andAngjoo
Kanazawa.2023.Nerfbusters:Removingghostlyartifactsfromcasuallycaptured InMILo,werelyontheDelaunaytriangulationasitnaturally
nerfs.InProceedingsoftheIEEE/CVFInternationalConferenceonComputerVision. adaptstothelocaldensityoftheinputpointcloud.Specifically,
18120û18130.
FrancisWilliams,JeromeParent-Levesque,DerekNowrouzezahrai,DanielePanozzo, itcreatessmallertetrahedrainregionswithhigherpointdensity
KwangMooYi,andAndreaTagliasacchi.2020.VoronoiNet:GeneralFunctional andlargeronesinsparserregions.Thisadaptivenaturemakesit
ApproximatorswithLocalSupport.In2020IEEE/CVFConferenceonComputerVision
moreefficientthanuniformgrid-basedapproachesforextracting
andPatternRecognitionWorkshops(CVPRW)(Seattle,WA,USA).IEEE,1069û1073.
doi:10.1109/CVPRW50498.2020.00140 surfacesfromnon-uniformpointdistributions,suchasthoseformed
QiWu,JanickMartinezEsturo,AshkanMirzaei,NicolasMoenne-Loccoz,andZan byoptimized3DGaussians.Furthermore,theregularityoftheDe-
Gojcic.2024.3DGUT:EnablingDistortedCamerasandSecondaryRaysinGaussian
launaystructurefacilitatesGPU-acceleratedmeshextractionvia
Splatting.arXivpreprintarXiv:2412.12507(2024).
JiaweiYang,MarcoPavone,andYueWang.2023.FreeNeRF:ImprovingFew-shotNeural themarchingtetrahedraalgorithm[DoiandKoide1991],whichis
RenderingwithFreeFrequencyRegularization.InProc.IEEEConf.onComputer thecornerstoneofourdifferentiablemeshextractionprocess.
VisionandPatternRecognition(CVPR).
LiorYariv,JiataoGu,YoniKasten,andYaronLipman.2021.Volumerenderingofneural
implicitsurfaces.(2021). B Experiments
L.Yariv,P.Hedman,C.Reiser,D.Verbin,P.P.Srinivasan,R.Szeliski,J.T.Barron,andB.
Mildenhall.2023.BakedSDF:MeshingNeuralSDFsforReal-TimeViewSynthesis. B.1 ImplementationDetails
ACMSIGGRAPHConferencePapers(2023).
MulinYu,TaoLu,LinningXu,LihanJiang,YuanboXiangli,andBoDai.2024b.GSDF: Optimization. Foroptimization,weusetheAdamoptimizerwith
3DGSMeetsSDFforImprovedRenderingandReconstruction.InNeurIPS. thesamelearningratesfortheGaussianparametersas[Zhangetal.
ZehaoYu,AnpeiChen,BinbinHuang,TorstenSattler,andAndreasGeiger.2024a. 2024].Weusealearningrateof0.025fortheSDFvalues.Wedensify
Mip-Splatting:Alias-free3DGaussianSplatting.InCVPR.19447û19456.
ZehaoYu,TorstenSattler,andAndreasGeiger.2024.GaussianOpacityFields:Efficient Gaussiansfor3,000iterationsusingaggressivedensification[Fang
AdaptiveSurfaceReconstructioninUnboundedScenes. ACMTransactionson andWang2024b],duringwhichweonlyrelyonourphotometric
Graphics(2024).
ZehaoYu,TorstenSattler,andAndreasGeiger.2024.GaussianOpacityFields:Efficient
loss.WeintroducethevolumetricrenderingregularizationlossL
N
AdaptiveSurfaceReconstructioninUnboundedScenes.ACMTOG(2024). afterdensification,atiteration3,000.WeletGaussianspopulatethe
BaowenZhang,ChuanFang,RakeshShrestha,YixunLiang,XiaoxiaoLong,andPing
sceneandrefinetheirparametersfor5,000additionaliterations,
Tan.2024. RaDe-GS:RasterizingDepthinGaussianSplatting. arXivpreprint
arXiv:2406.01467(2024). duringwhichwerelyonourvolumetricrenderinglossL vol .After
WenyuanZhang,Yu-ShenLiu,andZhizhongHan.2024b. NeuralSignedDistance thispointwestopthedensificationandpruningprocedures.Thus,
FunctionInferencethroughSplatting3DGaussiansPulledonZero-LevelSet.In
ourmethoddoesnotintroduceadditionalsensitivitytoinitialization
AdvancesinNeuralInformationProcessingSystems.
ZiyuZhang,BinbinHuang,HanqingJiang,LiyangZhou,XiaojunXiang,andShun- beyondwhatisalreadypresentinstandard3DGSpipelines.
hanShen.2024a.QuadraticGaussianSplattingforEfficientandDetailedSurface Weintroduceourmeshextractionpipelineatiteration8,000.We
Reconstruction.arXivpreprintarXiv:2411.16392(2024).
extractameshateveryiterationandapplyourfulllossLtoour
representation.Themesh-in-the-loopoptimizationcontinuesforan
additional10,000iterations,foratotalof18,000iterations.Forour
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

1:14 ò AntoineGuΘdon,DiegoGomez,NissimMaruani,BingchenGong,GeorgeDrettakis,andMaksOvsjanikov
Table 6. Quantitative results for novel view synthesis on MipN- SDFnormalization. Inpractice,formorestableoptimization,we
eRF360[Barronetal.2022].WereportPSNR,SSIM,andLPIPS.Our optimizeTruncatedSDFvaluesnormalizedtobeintherange[?1,1]
methodmaintainscompetitiverenderingqualitywhilesignificantlyimprov-
usingatanhfunction.WhenintroducingourMesh-in-the-Loopop-
ingsurfacereconstruction.
|     |     |     |     | timization, | the initial SDF values | are computed | using a custom, |
| --- | --- | --- | --- | ----------- | ---------------------- | ------------ | --------------- |
IndoorScenes OutdoorScenes scalabledepth-fusionalgorithmthatoperatesdirectlyonourDe-
PSNR? SSIM? LPIPS? PSNR? SSIM? LPIPS? launaysitesratherthanonaregulargrid:ForeachDelaunaysite,
| 3DGS | 30.41 0.920 | 0.189 24.64 | 0.731 | 0.234 |     |     |     |
| ---- | ----------- | ----------- | ----- | ----- | --- | --- | --- |
wecomputethesigneddifferencebetweenthedepthmaprendered
| Mip-Splatting | 30.90 0.921 | 0.194 24.65 | 0.729 | 0.245 |     |     |     |
| ------------- | ----------- | ----------- | ----- | ----- | --- | --- | --- |
fromtheGaussiansandthedepthofthesite,andfusethesedis-
| BakedSDF | 27.06 0.836 | 0.258 22.47 | 0.585 | 0.349 |     |     |     |
| -------- | ----------- | ----------- | ----- | ----- | --- | --- | --- |
tancesbyaveragingacrossallviews.Thisinitializationprovides
| SuGaR | 29.43 0.906 | 0.225 22.93 | 0.629 | 0.356 |     |     |     |
| ----- | ----------- | ----------- | ----- | ----- | --- | --- | --- |
agoodstartingpointfortheSDFoptimization;Whilebeingover-
| 2DGS | 30.40 0.916 | 0.195 24.34 | 0.717 | 0.246 |     |     |     |
| ---- | ----------- | ----------- | ----- | ----- | --- | --- | --- |
smoothedandmissingmanyfinedetails,theinitialmeshisalreadya
| GOF | 30.79 0.924 | 0.184 24.82 | 0.750 | 0.202 |     |     |     |
| --- | ----------- | ----------- | ----- | ----- | --- | --- | --- |
reasonableapproximationofthesurfaceencodedbytheGaussians.
| RaDe-GS    | 30.74 0.928 | 0.165 25.17 | 0.764 | 0.199 |     |     |     |
| ---------- | ----------- | ----------- | ----- | ----- | --- | --- | --- |
| Ours(base) | 29.96 0.920 | 0.191 24.47 | 0.718 | 0.290 |     |     |     |
Ours(dense) 30.76 0.934 0.155 24.81 0.744 0.229 Meshrendering. Fordifferentiablemeshrendering,weusenvd-
iffrast[Laineetal.2020],whichprovidesefficientGPU-accelerated
rasterizationwithgradientsupport.Werenderthemeshatthesame
basemodel,wepruneGaussiansatiteration8,000withimportance-
resolutionastheGaussianrendering,usingthesamecameraparam-
weightedsampling[FangandWang2024b]tomaintainonlythe
mostimportantGaussians.Forthedensemodel,wemaintainalarger eters.Tosmoothoutdiscontinuitiesandensurethepropagationof
setofGaussiansbutcomputetheimportancescoresatiteration8,000 gradientsbetweenneighborpixels,weapplyantialiasingsmoothing
andsampleDelaunaysitesonlyfromthemostimportantGaussians. tothedepthmapsrenderedfromthemesh.
| Ourweightparametersare? |            | = 0.2,? | =? =? | = 0.05,                          |     |     |     |
| ----------------------- | ---------- | ------- | ----- | -------------------------------- | --- | --- | --- |
|                         |            | RGB N   | MD MN | B.2 Mesh-BasedNovelViewSynthesis |     |     |     |
| and?                    | =? =0.005. |         |       |                                  |     |     |     |
| erosion                 | interior   |         |       |                                  |     |     |     |
Weshowcaseaqualitativecomparisonofourmesh-basednovel
| Delaunaytriangulation. |     | FortheDelaunaytriangulation,weuse |     |     |     |     |     |
| ---------------------- | --- | --------------------------------- | --- | --- | --- | --- | --- |
viewsynthesisevaluationinFig.10.
| the CGAL | libraryÆs 3D | Delaunay triangulation | implementation, |     |     |     |     |
| -------- | ------------ | ---------------------- | --------------- | --- | --- | --- | --- |
which provides robust and efficient computation of the tetrahe- B.3 NovelViewSynthesis
| dralization. | Although Delaunay | triangulation | is inherently | non- |     |     |     |
| ------------ | ----------------- | ------------- | ------------- | ---- | --- | --- | --- |
Whileourprimaryfocusisonsurfacereconstruction,wealsoeval-
differentiable,thisdoesnotposeaprobleminoursetting,asgradi-
uatethenovelviewsynthesisqualityofouroptimizedGaussians.
entspropagatefromthemeshbacktotheGaussiansviatheGaussian
Table6presentsthePSNR,SSIM,andLPIPSmetricsontheMipN-
pivots.WefoundthatupdatingtheDelaunaytriangulationatevery
eRF360[Barronetal.2022]dataset.Ourmethodmaintainscompet-
iterationisnotnecessaryforstableoptimization;Wethereforeonly
itiverenderingqualitycomparedtopreviousapproaches,demon-
updatetheDelaunaytriangulationevery500iterations.
stratingthatourmesh-in-the-loopoptimizationdoesnotcompro-
misethevisualfidelityoftheGaussianrepresentation.
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.

MILo:Mesh-In-the-LoopGaussianSplattingforDetailedandEfficientSurfaceReconstruction ò 1:15
Fig.10. Ourmesh-basedevaluationservesasaproxyforquantitativelyassessinggeometricartifacts,meshcompleteness,andbackground
reconstruction.MILodemonstratessuperiorperformanceincapturingfinedetailsùsuchastheintricatechandelierstemandchairspindlesùand
consistentlyrecovershigh-qualitybackgroundgeometry.
ACMTrans.Graph.,Vol.44,No.6,Article1.Publicationdate:December2025.
