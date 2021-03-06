??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??	
?
conv2d_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_80/kernel
}
$conv2d_80/kernel/Read/ReadVariableOpReadVariableOpconv2d_80/kernel*&
_output_shapes
:*
dtype0
t
conv2d_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_80/bias
m
"conv2d_80/bias/Read/ReadVariableOpReadVariableOpconv2d_80/bias*
_output_shapes
:*
dtype0
?
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_28/gamma
?
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_28/beta
?
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_28/moving_mean
?
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_28/moving_variance
?
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_81/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_81/kernel
}
$conv2d_81/kernel/Read/ReadVariableOpReadVariableOpconv2d_81/kernel*&
_output_shapes
: *
dtype0
t
conv2d_81/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_81/bias
m
"conv2d_81/bias/Read/ReadVariableOpReadVariableOpconv2d_81/bias*
_output_shapes
: *
dtype0
?
batch_normalization_29/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_29/gamma
?
0batch_normalization_29/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_29/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_29/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_29/beta
?
/batch_normalization_29/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_29/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_29/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_29/moving_mean
?
6batch_normalization_29/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_29/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_29/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_29/moving_variance
?
:batch_normalization_29/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_29/moving_variance*
_output_shapes
: *
dtype0
{
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_57/kernel
t
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes
:	?*
dtype0
r
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_57/bias
k
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes
:*
dtype0
j
Adam_1/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdam_1/iter
c
Adam_1/iter/Read/ReadVariableOpReadVariableOpAdam_1/iter*
_output_shapes
: *
dtype0	
n
Adam_1/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam_1/beta_1
g
!Adam_1/beta_1/Read/ReadVariableOpReadVariableOpAdam_1/beta_1*
_output_shapes
: *
dtype0
n
Adam_1/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam_1/beta_2
g
!Adam_1/beta_2/Read/ReadVariableOpReadVariableOpAdam_1/beta_2*
_output_shapes
: *
dtype0
l
Adam_1/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam_1/decay
e
 Adam_1/decay/Read/ReadVariableOpReadVariableOpAdam_1/decay*
_output_shapes
: *
dtype0
|
Adam_1/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam_1/learning_rate
u
(Adam_1/learning_rate/Read/ReadVariableOpReadVariableOpAdam_1/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam_1/conv2d_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam_1/conv2d_80/kernel/m
?
-Adam_1/conv2d_80/kernel/m/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_80/kernel/m*&
_output_shapes
:*
dtype0
?
Adam_1/conv2d_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam_1/conv2d_80/bias/m

+Adam_1/conv2d_80/bias/m/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_80/bias/m*
_output_shapes
:*
dtype0
?
%Adam_1/batch_normalization_28/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam_1/batch_normalization_28/gamma/m
?
9Adam_1/batch_normalization_28/gamma/m/Read/ReadVariableOpReadVariableOp%Adam_1/batch_normalization_28/gamma/m*
_output_shapes
:*
dtype0
?
$Adam_1/batch_normalization_28/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam_1/batch_normalization_28/beta/m
?
8Adam_1/batch_normalization_28/beta/m/Read/ReadVariableOpReadVariableOp$Adam_1/batch_normalization_28/beta/m*
_output_shapes
:*
dtype0
?
Adam_1/conv2d_81/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam_1/conv2d_81/kernel/m
?
-Adam_1/conv2d_81/kernel/m/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_81/kernel/m*&
_output_shapes
: *
dtype0
?
Adam_1/conv2d_81/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam_1/conv2d_81/bias/m

+Adam_1/conv2d_81/bias/m/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_81/bias/m*
_output_shapes
: *
dtype0
?
%Adam_1/batch_normalization_29/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam_1/batch_normalization_29/gamma/m
?
9Adam_1/batch_normalization_29/gamma/m/Read/ReadVariableOpReadVariableOp%Adam_1/batch_normalization_29/gamma/m*
_output_shapes
: *
dtype0
?
$Adam_1/batch_normalization_29/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam_1/batch_normalization_29/beta/m
?
8Adam_1/batch_normalization_29/beta/m/Read/ReadVariableOpReadVariableOp$Adam_1/batch_normalization_29/beta/m*
_output_shapes
: *
dtype0
?
Adam_1/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdam_1/dense_57/kernel/m
?
,Adam_1/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam_1/dense_57/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam_1/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam_1/dense_57/bias/m
}
*Adam_1/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam_1/dense_57/bias/m*
_output_shapes
:*
dtype0
?
Adam_1/conv2d_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam_1/conv2d_80/kernel/v
?
-Adam_1/conv2d_80/kernel/v/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_80/kernel/v*&
_output_shapes
:*
dtype0
?
Adam_1/conv2d_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam_1/conv2d_80/bias/v

+Adam_1/conv2d_80/bias/v/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_80/bias/v*
_output_shapes
:*
dtype0
?
%Adam_1/batch_normalization_28/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam_1/batch_normalization_28/gamma/v
?
9Adam_1/batch_normalization_28/gamma/v/Read/ReadVariableOpReadVariableOp%Adam_1/batch_normalization_28/gamma/v*
_output_shapes
:*
dtype0
?
$Adam_1/batch_normalization_28/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam_1/batch_normalization_28/beta/v
?
8Adam_1/batch_normalization_28/beta/v/Read/ReadVariableOpReadVariableOp$Adam_1/batch_normalization_28/beta/v*
_output_shapes
:*
dtype0
?
Adam_1/conv2d_81/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam_1/conv2d_81/kernel/v
?
-Adam_1/conv2d_81/kernel/v/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_81/kernel/v*&
_output_shapes
: *
dtype0
?
Adam_1/conv2d_81/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam_1/conv2d_81/bias/v

+Adam_1/conv2d_81/bias/v/Read/ReadVariableOpReadVariableOpAdam_1/conv2d_81/bias/v*
_output_shapes
: *
dtype0
?
%Adam_1/batch_normalization_29/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam_1/batch_normalization_29/gamma/v
?
9Adam_1/batch_normalization_29/gamma/v/Read/ReadVariableOpReadVariableOp%Adam_1/batch_normalization_29/gamma/v*
_output_shapes
: *
dtype0
?
$Adam_1/batch_normalization_29/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam_1/batch_normalization_29/beta/v
?
8Adam_1/batch_normalization_29/beta/v/Read/ReadVariableOpReadVariableOp$Adam_1/batch_normalization_29/beta/v*
_output_shapes
: *
dtype0
?
Adam_1/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdam_1/dense_57/kernel/v
?
,Adam_1/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam_1/dense_57/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam_1/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam_1/dense_57/bias/v
}
*Adam_1/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam_1/dense_57/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?W
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?V
value?VB?V B?V
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

_init_input_shape* 
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
axis
	gamma
beta
 moving_mean
!moving_variance
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
?

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
?
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses* 
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
?

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?m?m?.m?/m?7m?8m?Sm?Tm?v?v?v?v?.v?/v?7v?8v?Sv?Tv?*
j
0
1
2
3
 4
!5
.6
/7
78
89
910
:11
S12
T13*
J
0
1
2
3
.4
/5
76
87
S8
T9*
* 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

eserving_default* 
* 
`Z
VARIABLE_VALUEconv2d_80/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_80/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_28/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_28/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_28/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_28/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
0
1
 2
!3*

0
1*
* 
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_81/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_81/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 
ke
VARIABLE_VALUEbatch_normalization_29/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_29/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_29/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_29/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
70
81
92
:3*

70
81*
* 
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_57/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

S0
T1*

S0
T1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 
NH
VARIABLE_VALUEAdam_1/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEAdam_1/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEAdam_1/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam_1/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam_1/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
 
 0
!1
92
:3*
J
0
1
2
3
4
5
6
7
	8

9*

?0*
* 
* 
* 
* 
* 
* 
* 
* 

 0
!1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

90
:1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
?
VARIABLE_VALUEAdam_1/conv2d_80/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam_1/conv2d_80/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam_1/batch_normalization_28/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam_1/batch_normalization_28/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam_1/conv2d_81/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam_1/conv2d_81/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam_1/batch_normalization_29/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam_1/batch_normalization_29/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam_1/dense_57/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam_1/dense_57/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam_1/conv2d_80/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam_1/conv2d_80/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam_1/batch_normalization_28/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam_1/batch_normalization_28/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam_1/conv2d_81/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam_1/conv2d_81/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam_1/batch_normalization_29/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE$Adam_1/batch_normalization_29/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam_1/dense_57/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam_1/dense_57/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_23Placeholder*/
_output_shapes
:?????????PP*
dtype0*$
shape:?????????PP
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_23conv2d_80/kernelconv2d_80/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_81/kernelconv2d_81/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_variancedense_57/kerneldense_57/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8? *.
f)R'
%__inference_signature_wrapper_1482573
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_80/kernel/Read/ReadVariableOp"conv2d_80/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOp$conv2d_81/kernel/Read/ReadVariableOp"conv2d_81/bias/Read/ReadVariableOp0batch_normalization_29/gamma/Read/ReadVariableOp/batch_normalization_29/beta/Read/ReadVariableOp6batch_normalization_29/moving_mean/Read/ReadVariableOp:batch_normalization_29/moving_variance/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOpAdam_1/iter/Read/ReadVariableOp!Adam_1/beta_1/Read/ReadVariableOp!Adam_1/beta_2/Read/ReadVariableOp Adam_1/decay/Read/ReadVariableOp(Adam_1/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam_1/conv2d_80/kernel/m/Read/ReadVariableOp+Adam_1/conv2d_80/bias/m/Read/ReadVariableOp9Adam_1/batch_normalization_28/gamma/m/Read/ReadVariableOp8Adam_1/batch_normalization_28/beta/m/Read/ReadVariableOp-Adam_1/conv2d_81/kernel/m/Read/ReadVariableOp+Adam_1/conv2d_81/bias/m/Read/ReadVariableOp9Adam_1/batch_normalization_29/gamma/m/Read/ReadVariableOp8Adam_1/batch_normalization_29/beta/m/Read/ReadVariableOp,Adam_1/dense_57/kernel/m/Read/ReadVariableOp*Adam_1/dense_57/bias/m/Read/ReadVariableOp-Adam_1/conv2d_80/kernel/v/Read/ReadVariableOp+Adam_1/conv2d_80/bias/v/Read/ReadVariableOp9Adam_1/batch_normalization_28/gamma/v/Read/ReadVariableOp8Adam_1/batch_normalization_28/beta/v/Read/ReadVariableOp-Adam_1/conv2d_81/kernel/v/Read/ReadVariableOp+Adam_1/conv2d_81/bias/v/Read/ReadVariableOp9Adam_1/batch_normalization_29/gamma/v/Read/ReadVariableOp8Adam_1/batch_normalization_29/beta/v/Read/ReadVariableOp,Adam_1/dense_57/kernel/v/Read/ReadVariableOp*Adam_1/dense_57/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *)
f$R"
 __inference__traced_save_1482942
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_80/kernelconv2d_80/biasbatch_normalization_28/gammabatch_normalization_28/beta"batch_normalization_28/moving_mean&batch_normalization_28/moving_varianceconv2d_81/kernelconv2d_81/biasbatch_normalization_29/gammabatch_normalization_29/beta"batch_normalization_29/moving_mean&batch_normalization_29/moving_variancedense_57/kerneldense_57/biasAdam_1/iterAdam_1/beta_1Adam_1/beta_2Adam_1/decayAdam_1/learning_ratetotalcountAdam_1/conv2d_80/kernel/mAdam_1/conv2d_80/bias/m%Adam_1/batch_normalization_28/gamma/m$Adam_1/batch_normalization_28/beta/mAdam_1/conv2d_81/kernel/mAdam_1/conv2d_81/bias/m%Adam_1/batch_normalization_29/gamma/m$Adam_1/batch_normalization_29/beta/mAdam_1/dense_57/kernel/mAdam_1/dense_57/bias/mAdam_1/conv2d_80/kernel/vAdam_1/conv2d_80/bias/v%Adam_1/batch_normalization_28/gamma/v$Adam_1/batch_normalization_28/beta/vAdam_1/conv2d_81/kernel/vAdam_1/conv2d_81/bias/v%Adam_1/batch_normalization_29/gamma/v$Adam_1/batch_normalization_29/beta/vAdam_1/dense_57/kernel/vAdam_1/dense_57/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *,
f'R%
#__inference__traced_restore_1483075Ԕ
?C
?
E__inference_model_38_layer_call_and_return_conditional_losses_1482482

inputsB
(conv2d_80_conv2d_readvariableop_resource:7
)conv2d_80_biasadd_readvariableop_resource:<
.batch_normalization_28_readvariableop_resource:>
0batch_normalization_28_readvariableop_1_resource:M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_81_conv2d_readvariableop_resource: 7
)conv2d_81_biasadd_readvariableop_resource: <
.batch_normalization_29_readvariableop_resource: >
0batch_normalization_29_readvariableop_1_resource: M
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource: :
'dense_57_matmul_readvariableop_resource:	?6
(dense_57_biasadd_readvariableop_resource:
identity??6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1? conv2d_80/BiasAdd/ReadVariableOp?conv2d_80/Conv2D/ReadVariableOp? conv2d_81/BiasAdd/ReadVariableOp?conv2d_81/Conv2D/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?
conv2d_80/Conv2D/ReadVariableOpReadVariableOp(conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_80/Conv2DConv2Dinputs'conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_80/BiasAdd/ReadVariableOpReadVariableOp)conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_80/BiasAddBiasAddconv2d_80/Conv2D:output:0(conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_80/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( |
re_lu_48/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:??????????
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_81/Conv2DConv2Dre_lu_48/Relu:activations:0'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingSAME*
strides
?
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 ?
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3conv2d_81/BiasAdd:output:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????		 : : : : :*
epsilon%o?:*
is_training( |
re_lu_49/ReluRelu+batch_normalization_29/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????		 ?
max_pooling2d_37/MaxPoolMaxPoolre_lu_49/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
a
flatten_32/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_32/ReshapeReshape!max_pooling2d_37/MaxPool:output:0flatten_32/Const:output:0*
T0*(
_output_shapes
:???????????
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_57/MatMulMatMulflatten_32/Reshape:output:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_57/SoftmaxSoftmaxdense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_57/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp7^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_17^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_1!^conv2d_80/BiasAdd/ReadVariableOp ^conv2d_80/Conv2D/ReadVariableOp!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 2p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12D
 conv2d_80/BiasAdd/ReadVariableOp conv2d_80/BiasAdd/ReadVariableOp2B
conv2d_80/Conv2D/ReadVariableOpconv2d_80/Conv2D/ReadVariableOp2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?
a
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482755

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????		 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????		 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		 :W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?,
?
E__inference_model_38_layer_call_and_return_conditional_losses_1482313
input_23+
conv2d_80_1482275:
conv2d_80_1482277:,
batch_normalization_28_1482280:,
batch_normalization_28_1482282:,
batch_normalization_28_1482284:,
batch_normalization_28_1482286:+
conv2d_81_1482290: 
conv2d_81_1482292: ,
batch_normalization_29_1482295: ,
batch_normalization_29_1482297: ,
batch_normalization_29_1482299: ,
batch_normalization_29_1482301: #
dense_57_1482307:	?
dense_57_1482309:
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCallinput_23conv2d_80_1482275conv2d_80_1482277*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1481972?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0batch_normalization_28_1482280batch_normalization_28_1482282batch_normalization_28_1482284batch_normalization_28_1482286*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481837?
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1481992?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_81_1482290conv2d_81_1482292*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482004?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0batch_normalization_29_1482295batch_normalization_29_1482297batch_normalization_29_1482299batch_normalization_29_1482301*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481901?
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482024?
 max_pooling2d_37/PartitionedCallPartitionedCall!re_lu_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1481952?
flatten_32/PartitionedCallPartitionedCall)max_pooling2d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482033?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_32/PartitionedCall:output:0dense_57_1482307dense_57_1482309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1482046x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????PP
"
_user_specified_name
input_23
?M
?
"__inference__wrapped_model_1481815
input_23K
1model_38_conv2d_80_conv2d_readvariableop_resource:@
2model_38_conv2d_80_biasadd_readvariableop_resource:E
7model_38_batch_normalization_28_readvariableop_resource:G
9model_38_batch_normalization_28_readvariableop_1_resource:V
Hmodel_38_batch_normalization_28_fusedbatchnormv3_readvariableop_resource:X
Jmodel_38_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:K
1model_38_conv2d_81_conv2d_readvariableop_resource: @
2model_38_conv2d_81_biasadd_readvariableop_resource: E
7model_38_batch_normalization_29_readvariableop_resource: G
9model_38_batch_normalization_29_readvariableop_1_resource: V
Hmodel_38_batch_normalization_29_fusedbatchnormv3_readvariableop_resource: X
Jmodel_38_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource: C
0model_38_dense_57_matmul_readvariableop_resource:	??
1model_38_dense_57_biasadd_readvariableop_resource:
identity???model_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?Amodel_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?.model_38/batch_normalization_28/ReadVariableOp?0model_38/batch_normalization_28/ReadVariableOp_1??model_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?Amodel_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?.model_38/batch_normalization_29/ReadVariableOp?0model_38/batch_normalization_29/ReadVariableOp_1?)model_38/conv2d_80/BiasAdd/ReadVariableOp?(model_38/conv2d_80/Conv2D/ReadVariableOp?)model_38/conv2d_81/BiasAdd/ReadVariableOp?(model_38/conv2d_81/Conv2D/ReadVariableOp?(model_38/dense_57/BiasAdd/ReadVariableOp?'model_38/dense_57/MatMul/ReadVariableOp?
(model_38/conv2d_80/Conv2D/ReadVariableOpReadVariableOp1model_38_conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_38/conv2d_80/Conv2DConv2Dinput_230model_38/conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)model_38/conv2d_80/BiasAdd/ReadVariableOpReadVariableOp2model_38_conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_38/conv2d_80/BiasAddBiasAdd"model_38/conv2d_80/Conv2D:output:01model_38/conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
.model_38/batch_normalization_28/ReadVariableOpReadVariableOp7model_38_batch_normalization_28_readvariableop_resource*
_output_shapes
:*
dtype0?
0model_38/batch_normalization_28/ReadVariableOp_1ReadVariableOp9model_38_batch_normalization_28_readvariableop_1_resource*
_output_shapes
:*
dtype0?
?model_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_38_batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_38_batch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
0model_38/batch_normalization_28/FusedBatchNormV3FusedBatchNormV3#model_38/conv2d_80/BiasAdd:output:06model_38/batch_normalization_28/ReadVariableOp:value:08model_38/batch_normalization_28/ReadVariableOp_1:value:0Gmodel_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0Imodel_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( ?
model_38/re_lu_48/ReluRelu4model_38/batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:??????????
(model_38/conv2d_81/Conv2D/ReadVariableOpReadVariableOp1model_38_conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
model_38/conv2d_81/Conv2DConv2D$model_38/re_lu_48/Relu:activations:00model_38/conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingSAME*
strides
?
)model_38/conv2d_81/BiasAdd/ReadVariableOpReadVariableOp2model_38_conv2d_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_38/conv2d_81/BiasAddBiasAdd"model_38/conv2d_81/Conv2D:output:01model_38/conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 ?
.model_38/batch_normalization_29/ReadVariableOpReadVariableOp7model_38_batch_normalization_29_readvariableop_resource*
_output_shapes
: *
dtype0?
0model_38/batch_normalization_29/ReadVariableOp_1ReadVariableOp9model_38_batch_normalization_29_readvariableop_1_resource*
_output_shapes
: *
dtype0?
?model_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOpHmodel_38_batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
Amodel_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmodel_38_batch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
0model_38/batch_normalization_29/FusedBatchNormV3FusedBatchNormV3#model_38/conv2d_81/BiasAdd:output:06model_38/batch_normalization_29/ReadVariableOp:value:08model_38/batch_normalization_29/ReadVariableOp_1:value:0Gmodel_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0Imodel_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????		 : : : : :*
epsilon%o?:*
is_training( ?
model_38/re_lu_49/ReluRelu4model_38/batch_normalization_29/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????		 ?
!model_38/max_pooling2d_37/MaxPoolMaxPool$model_38/re_lu_49/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
j
model_38/flatten_32/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
model_38/flatten_32/ReshapeReshape*model_38/max_pooling2d_37/MaxPool:output:0"model_38/flatten_32/Const:output:0*
T0*(
_output_shapes
:???????????
'model_38/dense_57/MatMul/ReadVariableOpReadVariableOp0model_38_dense_57_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_38/dense_57/MatMulMatMul$model_38/flatten_32/Reshape:output:0/model_38/dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(model_38/dense_57/BiasAdd/ReadVariableOpReadVariableOp1model_38_dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_38/dense_57/BiasAddBiasAdd"model_38/dense_57/MatMul:product:00model_38/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
model_38/dense_57/SoftmaxSoftmax"model_38/dense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????r
IdentityIdentity#model_38/dense_57/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp@^model_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOpB^model_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1/^model_38/batch_normalization_28/ReadVariableOp1^model_38/batch_normalization_28/ReadVariableOp_1@^model_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOpB^model_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1/^model_38/batch_normalization_29/ReadVariableOp1^model_38/batch_normalization_29/ReadVariableOp_1*^model_38/conv2d_80/BiasAdd/ReadVariableOp)^model_38/conv2d_80/Conv2D/ReadVariableOp*^model_38/conv2d_81/BiasAdd/ReadVariableOp)^model_38/conv2d_81/Conv2D/ReadVariableOp)^model_38/dense_57/BiasAdd/ReadVariableOp(^model_38/dense_57/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 2?
?model_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp?model_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp2?
Amodel_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1Amodel_38/batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12`
.model_38/batch_normalization_28/ReadVariableOp.model_38/batch_normalization_28/ReadVariableOp2d
0model_38/batch_normalization_28/ReadVariableOp_10model_38/batch_normalization_28/ReadVariableOp_12?
?model_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp?model_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp2?
Amodel_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1Amodel_38/batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12`
.model_38/batch_normalization_29/ReadVariableOp.model_38/batch_normalization_29/ReadVariableOp2d
0model_38/batch_normalization_29/ReadVariableOp_10model_38/batch_normalization_29/ReadVariableOp_12V
)model_38/conv2d_80/BiasAdd/ReadVariableOp)model_38/conv2d_80/BiasAdd/ReadVariableOp2T
(model_38/conv2d_80/Conv2D/ReadVariableOp(model_38/conv2d_80/Conv2D/ReadVariableOp2V
)model_38/conv2d_81/BiasAdd/ReadVariableOp)model_38/conv2d_81/BiasAdd/ReadVariableOp2T
(model_38/conv2d_81/Conv2D/ReadVariableOp(model_38/conv2d_81/Conv2D/ReadVariableOp2T
(model_38/dense_57/BiasAdd/ReadVariableOp(model_38/dense_57/BiasAdd/ReadVariableOp2R
'model_38/dense_57/MatMul/ReadVariableOp'model_38/dense_57/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????PP
"
_user_specified_name
input_23
?
?
*__inference_model_38_layer_call_fn_1482272
input_23!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_38_layer_call_and_return_conditional_losses_1482208o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????PP
"
_user_specified_name
input_23
?
i
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1482765

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_32_layer_call_fn_1482770

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482033a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_model_38_layer_call_fn_1482426

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_38_layer_call_and_return_conditional_losses_1482208o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481932

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1482745

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481868

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?,
?
E__inference_model_38_layer_call_and_return_conditional_losses_1482208

inputs+
conv2d_80_1482170:
conv2d_80_1482172:,
batch_normalization_28_1482175:,
batch_normalization_28_1482177:,
batch_normalization_28_1482179:,
batch_normalization_28_1482181:+
conv2d_81_1482185: 
conv2d_81_1482187: ,
batch_normalization_29_1482190: ,
batch_normalization_29_1482192: ,
batch_normalization_29_1482194: ,
batch_normalization_29_1482196: #
dense_57_1482202:	?
dense_57_1482204:
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_80_1482170conv2d_80_1482172*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1481972?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0batch_normalization_28_1482175batch_normalization_28_1482177batch_normalization_28_1482179batch_normalization_28_1482181*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481868?
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1481992?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_81_1482185conv2d_81_1482187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482004?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0batch_normalization_29_1482190batch_normalization_29_1482192batch_normalization_29_1482194batch_normalization_29_1482196*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481932?
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482024?
 max_pooling2d_37/PartitionedCallPartitionedCall!re_lu_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1481952?
flatten_32/PartitionedCallPartitionedCall)max_pooling2d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482033?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_32/PartitionedCall:output:0dense_57_1482202dense_57_1482204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1482046x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?,
?
E__inference_model_38_layer_call_and_return_conditional_losses_1482354
input_23+
conv2d_80_1482316:
conv2d_80_1482318:,
batch_normalization_28_1482321:,
batch_normalization_28_1482323:,
batch_normalization_28_1482325:,
batch_normalization_28_1482327:+
conv2d_81_1482331: 
conv2d_81_1482333: ,
batch_normalization_29_1482336: ,
batch_normalization_29_1482338: ,
batch_normalization_29_1482340: ,
batch_normalization_29_1482342: #
dense_57_1482348:	?
dense_57_1482350:
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCallinput_23conv2d_80_1482316conv2d_80_1482318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1481972?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0batch_normalization_28_1482321batch_normalization_28_1482323batch_normalization_28_1482325batch_normalization_28_1482327*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481868?
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1481992?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_81_1482331conv2d_81_1482333*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482004?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0batch_normalization_29_1482336batch_normalization_29_1482338batch_normalization_29_1482340batch_normalization_29_1482342*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481932?
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482024?
 max_pooling2d_37/PartitionedCallPartitionedCall!re_lu_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1481952?
flatten_32/PartitionedCallPartitionedCall)max_pooling2d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482033?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_32/PartitionedCall:output:0dense_57_1482348dense_57_1482350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1482046x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:Y U
/
_output_shapes
:?????????PP
"
_user_specified_name
input_23
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1482636

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1482592

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?
?
+__inference_conv2d_80_layer_call_fn_1482582

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1481972w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PP: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_29_layer_call_fn_1482696

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481901?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_37_layer_call_fn_1482760

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1481952?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482004

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????		 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_28_layer_call_fn_1482618

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481868?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_1483075
file_prefix;
!assignvariableop_conv2d_80_kernel:/
!assignvariableop_1_conv2d_80_bias:=
/assignvariableop_2_batch_normalization_28_gamma:<
.assignvariableop_3_batch_normalization_28_beta:C
5assignvariableop_4_batch_normalization_28_moving_mean:G
9assignvariableop_5_batch_normalization_28_moving_variance:=
#assignvariableop_6_conv2d_81_kernel: /
!assignvariableop_7_conv2d_81_bias: =
/assignvariableop_8_batch_normalization_29_gamma: <
.assignvariableop_9_batch_normalization_29_beta: D
6assignvariableop_10_batch_normalization_29_moving_mean: H
:assignvariableop_11_batch_normalization_29_moving_variance: 6
#assignvariableop_12_dense_57_kernel:	?/
!assignvariableop_13_dense_57_bias:)
assignvariableop_14_adam_1_iter:	 +
!assignvariableop_15_adam_1_beta_1: +
!assignvariableop_16_adam_1_beta_2: *
 assignvariableop_17_adam_1_decay: 2
(assignvariableop_18_adam_1_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: G
-assignvariableop_21_adam_1_conv2d_80_kernel_m:9
+assignvariableop_22_adam_1_conv2d_80_bias_m:G
9assignvariableop_23_adam_1_batch_normalization_28_gamma_m:F
8assignvariableop_24_adam_1_batch_normalization_28_beta_m:G
-assignvariableop_25_adam_1_conv2d_81_kernel_m: 9
+assignvariableop_26_adam_1_conv2d_81_bias_m: G
9assignvariableop_27_adam_1_batch_normalization_29_gamma_m: F
8assignvariableop_28_adam_1_batch_normalization_29_beta_m: ?
,assignvariableop_29_adam_1_dense_57_kernel_m:	?8
*assignvariableop_30_adam_1_dense_57_bias_m:G
-assignvariableop_31_adam_1_conv2d_80_kernel_v:9
+assignvariableop_32_adam_1_conv2d_80_bias_v:G
9assignvariableop_33_adam_1_batch_normalization_28_gamma_v:F
8assignvariableop_34_adam_1_batch_normalization_28_beta_v:G
-assignvariableop_35_adam_1_conv2d_81_kernel_v: 9
+assignvariableop_36_adam_1_conv2d_81_bias_v: G
9assignvariableop_37_adam_1_batch_normalization_29_gamma_v: F
8assignvariableop_38_adam_1_batch_normalization_29_beta_v: ?
,assignvariableop_39_adam_1_dense_57_kernel_v:	?8
*assignvariableop_40_adam_1_dense_57_bias_v:
identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_80_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_80_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_28_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_28_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_28_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_28_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_81_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_81_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_29_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_29_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_29_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_29_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_57_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_57_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_1_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_adam_1_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_adam_1_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp assignvariableop_17_adam_1_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_1_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_1_conv2d_80_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_1_conv2d_80_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp9assignvariableop_23_adam_1_batch_normalization_28_gamma_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp8assignvariableop_24_adam_1_batch_normalization_28_beta_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_1_conv2d_81_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_1_conv2d_81_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp9assignvariableop_27_adam_1_batch_normalization_29_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp8assignvariableop_28_adam_1_batch_normalization_29_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_1_dense_57_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_1_dense_57_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_1_conv2d_80_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_1_conv2d_80_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp9assignvariableop_33_adam_1_batch_normalization_28_gamma_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp8assignvariableop_34_adam_1_batch_normalization_28_beta_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_1_conv2d_81_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_1_conv2d_81_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp9assignvariableop_37_adam_1_batch_normalization_29_gamma_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp8assignvariableop_38_adam_1_batch_normalization_29_beta_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_1_dense_57_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_1_dense_57_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
8__inference_batch_normalization_29_layer_call_fn_1482709

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481932?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
F
*__inference_re_lu_49_layer_call_fn_1482750

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482024h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????		 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		 :W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?
c
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482776

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_dense_57_layer_call_and_return_conditional_losses_1482796

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_81_layer_call_fn_1482673

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482004w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????		 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1481972

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????PP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1482573
input_23!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8? *+
f&R$
"__inference__wrapped_model_1481815o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????PP
"
_user_specified_name
input_23
?,
?
E__inference_model_38_layer_call_and_return_conditional_losses_1482053

inputs+
conv2d_80_1481973:
conv2d_80_1481975:,
batch_normalization_28_1481978:,
batch_normalization_28_1481980:,
batch_normalization_28_1481982:,
batch_normalization_28_1481984:+
conv2d_81_1482005: 
conv2d_81_1482007: ,
batch_normalization_29_1482010: ,
batch_normalization_29_1482012: ,
batch_normalization_29_1482014: ,
batch_normalization_29_1482016: #
dense_57_1482047:	?
dense_57_1482049:
identity??.batch_normalization_28/StatefulPartitionedCall?.batch_normalization_29/StatefulPartitionedCall?!conv2d_80/StatefulPartitionedCall?!conv2d_81/StatefulPartitionedCall? dense_57/StatefulPartitionedCall?
!conv2d_80/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_80_1481973conv2d_80_1481975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1481972?
.batch_normalization_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_80/StatefulPartitionedCall:output:0batch_normalization_28_1481978batch_normalization_28_1481980batch_normalization_28_1481982batch_normalization_28_1481984*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481837?
re_lu_48/PartitionedCallPartitionedCall7batch_normalization_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1481992?
!conv2d_81/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_81_1482005conv2d_81_1482007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *O
fJRH
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482004?
.batch_normalization_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_81/StatefulPartitionedCall:output:0batch_normalization_29_1482010batch_normalization_29_1482012batch_normalization_29_1482014batch_normalization_29_1482016*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 *&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481901?
re_lu_49/PartitionedCallPartitionedCall7batch_normalization_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		 * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482024?
 max_pooling2d_37/PartitionedCallPartitionedCall!re_lu_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1481952?
flatten_32/PartitionedCallPartitionedCall)max_pooling2d_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *P
fKRI
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482033?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall#flatten_32/PartitionedCall:output:0dense_57_1482047dense_57_1482049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1482046x
IdentityIdentity)dense_57/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_28/StatefulPartitionedCall/^batch_normalization_29/StatefulPartitionedCall"^conv2d_80/StatefulPartitionedCall"^conv2d_81/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 2`
.batch_normalization_28/StatefulPartitionedCall.batch_normalization_28/StatefulPartitionedCall2`
.batch_normalization_29/StatefulPartitionedCall.batch_normalization_29/StatefulPartitionedCall2F
!conv2d_80/StatefulPartitionedCall!conv2d_80/StatefulPartitionedCall2F
!conv2d_81/StatefulPartitionedCall!conv2d_81/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?Q
?
E__inference_model_38_layer_call_and_return_conditional_losses_1482538

inputsB
(conv2d_80_conv2d_readvariableop_resource:7
)conv2d_80_biasadd_readvariableop_resource:<
.batch_normalization_28_readvariableop_resource:>
0batch_normalization_28_readvariableop_1_resource:M
?batch_normalization_28_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_81_conv2d_readvariableop_resource: 7
)conv2d_81_biasadd_readvariableop_resource: <
.batch_normalization_29_readvariableop_resource: >
0batch_normalization_29_readvariableop_1_resource: M
?batch_normalization_29_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource: :
'dense_57_matmul_readvariableop_resource:	?6
(dense_57_biasadd_readvariableop_resource:
identity??%batch_normalization_28/AssignNewValue?'batch_normalization_28/AssignNewValue_1?6batch_normalization_28/FusedBatchNormV3/ReadVariableOp?8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_28/ReadVariableOp?'batch_normalization_28/ReadVariableOp_1?%batch_normalization_29/AssignNewValue?'batch_normalization_29/AssignNewValue_1?6batch_normalization_29/FusedBatchNormV3/ReadVariableOp?8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_29/ReadVariableOp?'batch_normalization_29/ReadVariableOp_1? conv2d_80/BiasAdd/ReadVariableOp?conv2d_80/Conv2D/ReadVariableOp? conv2d_81/BiasAdd/ReadVariableOp?conv2d_81/Conv2D/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?
conv2d_80/Conv2D/ReadVariableOpReadVariableOp(conv2d_80_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_80/Conv2DConv2Dinputs'conv2d_80/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
 conv2d_80/BiasAdd/ReadVariableOpReadVariableOp)conv2d_80_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_80/BiasAddBiasAddconv2d_80/Conv2D:output:0(conv2d_80/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
%batch_normalization_28/ReadVariableOpReadVariableOp.batch_normalization_28_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_28/ReadVariableOp_1ReadVariableOp0batch_normalization_28_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_28/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_28/FusedBatchNormV3FusedBatchNormV3conv2d_80/BiasAdd:output:0-batch_normalization_28/ReadVariableOp:value:0/batch_normalization_28/ReadVariableOp_1:value:0>batch_normalization_28/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_28/AssignNewValueAssignVariableOp?batch_normalization_28_fusedbatchnormv3_readvariableop_resource4batch_normalization_28/FusedBatchNormV3:batch_mean:07^batch_normalization_28/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_28/AssignNewValue_1AssignVariableOpAbatch_normalization_28_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_28/FusedBatchNormV3:batch_variance:09^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0|
re_lu_48/ReluRelu+batch_normalization_28/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:??????????
conv2d_81/Conv2D/ReadVariableOpReadVariableOp(conv2d_81_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_81/Conv2DConv2Dre_lu_48/Relu:activations:0'conv2d_81/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingSAME*
strides
?
 conv2d_81/BiasAdd/ReadVariableOpReadVariableOp)conv2d_81_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_81/BiasAddBiasAddconv2d_81/Conv2D:output:0(conv2d_81/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 ?
%batch_normalization_29/ReadVariableOpReadVariableOp.batch_normalization_29_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_29/ReadVariableOp_1ReadVariableOp0batch_normalization_29_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_29/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_29/FusedBatchNormV3FusedBatchNormV3conv2d_81/BiasAdd:output:0-batch_normalization_29/ReadVariableOp:value:0/batch_normalization_29/ReadVariableOp_1:value:0>batch_normalization_29/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????		 : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_29/AssignNewValueAssignVariableOp?batch_normalization_29_fusedbatchnormv3_readvariableop_resource4batch_normalization_29/FusedBatchNormV3:batch_mean:07^batch_normalization_29/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_29/AssignNewValue_1AssignVariableOpAbatch_normalization_29_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_29/FusedBatchNormV3:batch_variance:09^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0|
re_lu_49/ReluRelu+batch_normalization_29/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????		 ?
max_pooling2d_37/MaxPoolMaxPoolre_lu_49/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
a
flatten_32/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_32/ReshapeReshape!max_pooling2d_37/MaxPool:output:0flatten_32/Const:output:0*
T0*(
_output_shapes
:???????????
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_57/MatMulMatMulflatten_32/Reshape:output:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_57/SoftmaxSoftmaxdense_57/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_57/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_28/AssignNewValue(^batch_normalization_28/AssignNewValue_17^batch_normalization_28/FusedBatchNormV3/ReadVariableOp9^batch_normalization_28/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_28/ReadVariableOp(^batch_normalization_28/ReadVariableOp_1&^batch_normalization_29/AssignNewValue(^batch_normalization_29/AssignNewValue_17^batch_normalization_29/FusedBatchNormV3/ReadVariableOp9^batch_normalization_29/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_29/ReadVariableOp(^batch_normalization_29/ReadVariableOp_1!^conv2d_80/BiasAdd/ReadVariableOp ^conv2d_80/Conv2D/ReadVariableOp!^conv2d_81/BiasAdd/ReadVariableOp ^conv2d_81/Conv2D/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 2N
%batch_normalization_28/AssignNewValue%batch_normalization_28/AssignNewValue2R
'batch_normalization_28/AssignNewValue_1'batch_normalization_28/AssignNewValue_12p
6batch_normalization_28/FusedBatchNormV3/ReadVariableOp6batch_normalization_28/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_28/FusedBatchNormV3/ReadVariableOp_18batch_normalization_28/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_28/ReadVariableOp%batch_normalization_28/ReadVariableOp2R
'batch_normalization_28/ReadVariableOp_1'batch_normalization_28/ReadVariableOp_12N
%batch_normalization_29/AssignNewValue%batch_normalization_29/AssignNewValue2R
'batch_normalization_29/AssignNewValue_1'batch_normalization_29/AssignNewValue_12p
6batch_normalization_29/FusedBatchNormV3/ReadVariableOp6batch_normalization_29/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_29/FusedBatchNormV3/ReadVariableOp_18batch_normalization_29/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_29/ReadVariableOp%batch_normalization_29/ReadVariableOp2R
'batch_normalization_29/ReadVariableOp_1'batch_normalization_29/ReadVariableOp_12D
 conv2d_80/BiasAdd/ReadVariableOp conv2d_80/BiasAdd/ReadVariableOp2B
conv2d_80/Conv2D/ReadVariableOpconv2d_80/Conv2D/ReadVariableOp2D
 conv2d_81/BiasAdd/ReadVariableOp conv2d_81/BiasAdd/ReadVariableOp2B
conv2d_81/Conv2D/ReadVariableOpconv2d_81/Conv2D/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481837

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1482654

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1481992

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482024

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????		 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????		 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		 :W S
/
_output_shapes
:?????????		 
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_28_layer_call_fn_1482605

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *\
fWRU
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1481837?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1482664

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482683

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????		 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?Y
?
 __inference__traced_save_1482942
file_prefix/
+savev2_conv2d_80_kernel_read_readvariableop-
)savev2_conv2d_80_bias_read_readvariableop;
7savev2_batch_normalization_28_gamma_read_readvariableop:
6savev2_batch_normalization_28_beta_read_readvariableopA
=savev2_batch_normalization_28_moving_mean_read_readvariableopE
Asavev2_batch_normalization_28_moving_variance_read_readvariableop/
+savev2_conv2d_81_kernel_read_readvariableop-
)savev2_conv2d_81_bias_read_readvariableop;
7savev2_batch_normalization_29_gamma_read_readvariableop:
6savev2_batch_normalization_29_beta_read_readvariableopA
=savev2_batch_normalization_29_moving_mean_read_readvariableopE
Asavev2_batch_normalization_29_moving_variance_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop*
&savev2_adam_1_iter_read_readvariableop	,
(savev2_adam_1_beta_1_read_readvariableop,
(savev2_adam_1_beta_2_read_readvariableop+
'savev2_adam_1_decay_read_readvariableop3
/savev2_adam_1_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_1_conv2d_80_kernel_m_read_readvariableop6
2savev2_adam_1_conv2d_80_bias_m_read_readvariableopD
@savev2_adam_1_batch_normalization_28_gamma_m_read_readvariableopC
?savev2_adam_1_batch_normalization_28_beta_m_read_readvariableop8
4savev2_adam_1_conv2d_81_kernel_m_read_readvariableop6
2savev2_adam_1_conv2d_81_bias_m_read_readvariableopD
@savev2_adam_1_batch_normalization_29_gamma_m_read_readvariableopC
?savev2_adam_1_batch_normalization_29_beta_m_read_readvariableop7
3savev2_adam_1_dense_57_kernel_m_read_readvariableop5
1savev2_adam_1_dense_57_bias_m_read_readvariableop8
4savev2_adam_1_conv2d_80_kernel_v_read_readvariableop6
2savev2_adam_1_conv2d_80_bias_v_read_readvariableopD
@savev2_adam_1_batch_normalization_28_gamma_v_read_readvariableopC
?savev2_adam_1_batch_normalization_28_beta_v_read_readvariableop8
4savev2_adam_1_conv2d_81_kernel_v_read_readvariableop6
2savev2_adam_1_conv2d_81_bias_v_read_readvariableopD
@savev2_adam_1_batch_normalization_29_gamma_v_read_readvariableopC
?savev2_adam_1_batch_normalization_29_beta_v_read_readvariableop7
3savev2_adam_1_dense_57_kernel_v_read_readvariableop5
1savev2_adam_1_dense_57_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_80_kernel_read_readvariableop)savev2_conv2d_80_bias_read_readvariableop7savev2_batch_normalization_28_gamma_read_readvariableop6savev2_batch_normalization_28_beta_read_readvariableop=savev2_batch_normalization_28_moving_mean_read_readvariableopAsavev2_batch_normalization_28_moving_variance_read_readvariableop+savev2_conv2d_81_kernel_read_readvariableop)savev2_conv2d_81_bias_read_readvariableop7savev2_batch_normalization_29_gamma_read_readvariableop6savev2_batch_normalization_29_beta_read_readvariableop=savev2_batch_normalization_29_moving_mean_read_readvariableopAsavev2_batch_normalization_29_moving_variance_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop&savev2_adam_1_iter_read_readvariableop(savev2_adam_1_beta_1_read_readvariableop(savev2_adam_1_beta_2_read_readvariableop'savev2_adam_1_decay_read_readvariableop/savev2_adam_1_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_1_conv2d_80_kernel_m_read_readvariableop2savev2_adam_1_conv2d_80_bias_m_read_readvariableop@savev2_adam_1_batch_normalization_28_gamma_m_read_readvariableop?savev2_adam_1_batch_normalization_28_beta_m_read_readvariableop4savev2_adam_1_conv2d_81_kernel_m_read_readvariableop2savev2_adam_1_conv2d_81_bias_m_read_readvariableop@savev2_adam_1_batch_normalization_29_gamma_m_read_readvariableop?savev2_adam_1_batch_normalization_29_beta_m_read_readvariableop3savev2_adam_1_dense_57_kernel_m_read_readvariableop1savev2_adam_1_dense_57_bias_m_read_readvariableop4savev2_adam_1_conv2d_80_kernel_v_read_readvariableop2savev2_adam_1_conv2d_80_bias_v_read_readvariableop@savev2_adam_1_batch_normalization_28_gamma_v_read_readvariableop?savev2_adam_1_batch_normalization_28_beta_v_read_readvariableop4savev2_adam_1_conv2d_81_kernel_v_read_readvariableop2savev2_adam_1_conv2d_81_bias_v_read_readvariableop@savev2_adam_1_batch_normalization_29_gamma_v_read_readvariableop?savev2_adam_1_batch_normalization_29_beta_v_read_readvariableop3savev2_adam_1_dense_57_kernel_v_read_readvariableop1savev2_adam_1_dense_57_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : : :	?:: : : : : : : ::::: : : : :	?:::::: : : : :	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: :%(!

_output_shapes
:	?: )

_output_shapes
::*

_output_shapes
: 
?
F
*__inference_re_lu_48_layer_call_fn_1482659

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1481992h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_model_38_layer_call_fn_1482084
input_23!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_23unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_38_layer_call_and_return_conditional_losses_1482053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????PP
"
_user_specified_name
input_23
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1482727

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1481901

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_dense_57_layer_call_fn_1482785

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_dense_57_layer_call_and_return_conditional_losses_1482046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482033

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_dense_57_layer_call_and_return_conditional_losses_1482046

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1481952

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_model_38_layer_call_fn_1482393

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:	?

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*1
config_proto!

CPU

GPU (2J 8? *N
fIRG
E__inference_model_38_layer_call_and_return_conditional_losses_1482053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????PP: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????PP
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_239
serving_default_input_23:0?????????PP<
dense_570
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
beta
 moving_mean
!moving_variance
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
?

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
?
6axis
	7gamma
8beta
9moving_mean
:moving_variance
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?m?m?.m?/m?7m?8m?Sm?Tm?v?v?v?v?.v?/v?7v?8v?Sv?Tv?"
	optimizer
?
0
1
2
3
 4
!5
.6
/7
78
89
910
:11
S12
T13"
trackable_list_wrapper
f
0
1
2
3
.4
/5
76
87
S8
T9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_model_38_layer_call_fn_1482084
*__inference_model_38_layer_call_fn_1482393
*__inference_model_38_layer_call_fn_1482426
*__inference_model_38_layer_call_fn_1482272?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_38_layer_call_and_return_conditional_losses_1482482
E__inference_model_38_layer_call_and_return_conditional_losses_1482538
E__inference_model_38_layer_call_and_return_conditional_losses_1482313
E__inference_model_38_layer_call_and_return_conditional_losses_1482354?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1481815input_23"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
eserving_default"
signature_map
 "
trackable_list_wrapper
*:(2conv2d_80/kernel
:2conv2d_80/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_conv2d_80_layer_call_fn_1482582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1482592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
*:(2batch_normalization_28/gamma
):'2batch_normalization_28/beta
2:0 (2"batch_normalization_28/moving_mean
6:4 (2&batch_normalization_28/moving_variance
<
0
1
 2
!3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_28_layer_call_fn_1482605
8__inference_batch_normalization_28_layer_call_fn_1482618?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1482636
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1482654?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_re_lu_48_layer_call_fn_1482659?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1482664?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
*:( 2conv2d_81/kernel
: 2conv2d_81/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_conv2d_81_layer_call_fn_1482673?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482683?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
*:( 2batch_normalization_29/gamma
):' 2batch_normalization_29/beta
2:0  (2"batch_normalization_29/moving_mean
6:4  (2&batch_normalization_29/moving_variance
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_29_layer_call_fn_1482696
8__inference_batch_normalization_29_layer_call_fn_1482709?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1482727
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1482745?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_re_lu_49_layer_call_fn_1482750?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482755?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_max_pooling2d_37_layer_call_fn_1482760?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1482765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_flatten_32_layer_call_fn_1482770?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482776?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": 	?2dense_57/kernel
:2dense_57/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_57_layer_call_fn_1482785?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_57_layer_call_and_return_conditional_losses_1482796?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2Adam_1/iter
: (2Adam_1/beta_1
: (2Adam_1/beta_2
: (2Adam_1/decay
: (2Adam_1/learning_rate
<
 0
!1
92
:3"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_1482573input_23"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
1:/2Adam_1/conv2d_80/kernel/m
#:!2Adam_1/conv2d_80/bias/m
1:/2%Adam_1/batch_normalization_28/gamma/m
0:.2$Adam_1/batch_normalization_28/beta/m
1:/ 2Adam_1/conv2d_81/kernel/m
#:! 2Adam_1/conv2d_81/bias/m
1:/ 2%Adam_1/batch_normalization_29/gamma/m
0:. 2$Adam_1/batch_normalization_29/beta/m
):'	?2Adam_1/dense_57/kernel/m
": 2Adam_1/dense_57/bias/m
1:/2Adam_1/conv2d_80/kernel/v
#:!2Adam_1/conv2d_80/bias/v
1:/2%Adam_1/batch_normalization_28/gamma/v
0:.2$Adam_1/batch_normalization_28/beta/v
1:/ 2Adam_1/conv2d_81/kernel/v
#:! 2Adam_1/conv2d_81/bias/v
1:/ 2%Adam_1/batch_normalization_29/gamma/v
0:. 2$Adam_1/batch_normalization_29/beta/v
):'	?2Adam_1/dense_57/kernel/v
": 2Adam_1/dense_57/bias/v?
"__inference__wrapped_model_1481815? !./789:ST9?6
/?,
*?'
input_23?????????PP
? "3?0
.
dense_57"?
dense_57??????????
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1482636? !M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_28_layer_call_and_return_conditional_losses_1482654? !M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
8__inference_batch_normalization_28_layer_call_fn_1482605? !M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_28_layer_call_fn_1482618? !M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1482727?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
S__inference_batch_normalization_29_layer_call_and_return_conditional_losses_1482745?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
8__inference_batch_normalization_29_layer_call_fn_1482696?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
8__inference_batch_normalization_29_layer_call_fn_1482709?789:M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
F__inference_conv2d_80_layer_call_and_return_conditional_losses_1482592l7?4
-?*
(?%
inputs?????????PP
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_80_layer_call_fn_1482582_7?4
-?*
(?%
inputs?????????PP
? " ???????????
F__inference_conv2d_81_layer_call_and_return_conditional_losses_1482683l./7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????		 
? ?
+__inference_conv2d_81_layer_call_fn_1482673_./7?4
-?*
(?%
inputs?????????
? " ??????????		 ?
E__inference_dense_57_layer_call_and_return_conditional_losses_1482796]ST0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_dense_57_layer_call_fn_1482785PST0?-
&?#
!?
inputs??????????
? "???????????
G__inference_flatten_32_layer_call_and_return_conditional_losses_1482776a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
,__inference_flatten_32_layer_call_fn_1482770T7?4
-?*
(?%
inputs????????? 
? "????????????
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_1482765?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_37_layer_call_fn_1482760?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_model_38_layer_call_and_return_conditional_losses_1482313z !./789:STA?>
7?4
*?'
input_23?????????PP
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_38_layer_call_and_return_conditional_losses_1482354z !./789:STA?>
7?4
*?'
input_23?????????PP
p

 
? "%?"
?
0?????????
? ?
E__inference_model_38_layer_call_and_return_conditional_losses_1482482x !./789:ST??<
5?2
(?%
inputs?????????PP
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_38_layer_call_and_return_conditional_losses_1482538x !./789:ST??<
5?2
(?%
inputs?????????PP
p

 
? "%?"
?
0?????????
? ?
*__inference_model_38_layer_call_fn_1482084m !./789:STA?>
7?4
*?'
input_23?????????PP
p 

 
? "???????????
*__inference_model_38_layer_call_fn_1482272m !./789:STA?>
7?4
*?'
input_23?????????PP
p

 
? "???????????
*__inference_model_38_layer_call_fn_1482393k !./789:ST??<
5?2
(?%
inputs?????????PP
p 

 
? "???????????
*__inference_model_38_layer_call_fn_1482426k !./789:ST??<
5?2
(?%
inputs?????????PP
p

 
? "???????????
E__inference_re_lu_48_layer_call_and_return_conditional_losses_1482664h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_re_lu_48_layer_call_fn_1482659[7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_re_lu_49_layer_call_and_return_conditional_losses_1482755h7?4
-?*
(?%
inputs?????????		 
? "-?*
#? 
0?????????		 
? ?
*__inference_re_lu_49_layer_call_fn_1482750[7?4
-?*
(?%
inputs?????????		 
? " ??????????		 ?
%__inference_signature_wrapper_1482573? !./789:STE?B
? 
;?8
6
input_23*?'
input_23?????????PP"3?0
.
dense_57"?
dense_57?????????