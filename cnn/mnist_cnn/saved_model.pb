ßý
§//
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ě
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
î
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.12.02
b'unknown'˛Ś
f
XPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
yPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

P
is_trainingPlaceholder*
dtype0
*
shape:*
_output_shapes
:
f
Reshape/shapeConst*%
valueB"˙˙˙˙         *
dtype0*
_output_shapes
:
l
ReshapeReshapeXReshape/shape*
T0*
Tshape0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
0conv2d/kernel/Initializer/truncated_normal/shapeConst*%
valueB"             *
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
:

/conv2d/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 

1conv2d/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *¸1	?*
dtype0* 
_class
loc:@conv2d/kernel*
_output_shapes
: 
ř
:conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal0conv2d/kernel/Initializer/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed * 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
÷
.conv2d/kernel/Initializer/truncated_normal/mulMul:conv2d/kernel/Initializer/truncated_normal/TruncatedNormal1conv2d/kernel/Initializer/truncated_normal/stddev*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
ĺ
*conv2d/kernel/Initializer/truncated_normalAdd.conv2d/kernel/Initializer/truncated_normal/mul/conv2d/kernel/Initializer/truncated_normal/mean*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
ł
conv2d/kernel
VariableV2*
dtype0*
shared_name *
shape: *
	container * 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
Ő
conv2d/kernel/AssignAssignconv2d/kernel*conv2d/kernel/Initializer/truncated_normal*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 

conv2d/kernel/readIdentityconv2d/kernel*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 

conv2d/bias/Initializer/zerosConst*
valueB *    *
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 

conv2d/bias
VariableV2*
dtype0*
shared_name *
shape: *
	container *
_class
loc:@conv2d/bias*
_output_shapes
: 
ś
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
n
conv2d/bias/readIdentityconv2d/bias*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ü
conv2d/Conv2DConv2DReshapeconv2d/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
]
conv2d/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ş
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ż
2conv2d_1/kernel/Initializer/truncated_normal/shapeConst*%
valueB"              *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:

1conv2d_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

3conv2d_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *Â=*
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
ţ
<conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2conv2d_1/kernel/Initializer/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
˙
0conv2d_1/kernel/Initializer/truncated_normal/mulMul<conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormal3conv2d_1/kernel/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
í
,conv2d_1/kernel/Initializer/truncated_normalAdd0conv2d_1/kernel/Initializer/truncated_normal/mul1conv2d_1/kernel/Initializer/truncated_normal/mean*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ˇ
conv2d_1/kernel
VariableV2*
dtype0*
shared_name *
shape:  *
	container *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
Ý
conv2d_1/kernel/AssignAssignconv2d_1/kernel,conv2d_1/kernel/Initializer/truncated_normal*
T0*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  

conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  

conv2d_1/bias/Initializer/zerosConst*
valueB *    *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 

conv2d_1/bias
VariableV2*
dtype0*
shared_name *
shape: *
	container * 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
ž
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
î
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
a
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ž
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
d
flatten/ShapeShapemax_pooling2d_1/MaxPool*
T0*
out_type0*
_output_shapes
:
e
flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ą
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
Index0*
end_mask *
shrink_axis_mask*
T0*

begin_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
: 
b
flatten/Reshape/shape/1Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:

flatten/ReshapeReshapemax_pooling2d_1/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ą
/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:

.dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 

0dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *ôM&=*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
ď
9dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal/dense/kernel/Initializer/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed *
_class
loc:@dense/kernel* 
_output_shapes
:
 
í
-dense/kernel/Initializer/truncated_normal/mulMul9dense/kernel/Initializer/truncated_normal/TruncatedNormal0dense/kernel/Initializer/truncated_normal/stddev*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
 
Ű
)dense/kernel/Initializer/truncated_normalAdd-dense/kernel/Initializer/truncated_normal/mul.dense/kernel/Initializer/truncated_normal/mean*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
 
Ľ
dense/kernel
VariableV2*
dtype0*
shared_name *
shape:
 *
	container *
_class
loc:@dense/kernel* 
_output_shapes
:
 
Ë
dense/kernel/AssignAssigndense/kernel)dense/kernel/Initializer/truncated_normal*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel* 
_output_shapes
:
 
w
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
 

,dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*
dtype0*
_class
loc:@dense/bias*
_output_shapes
:

"dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_class
loc:@dense/bias*
_output_shapes
: 
Í
dense/bias/Initializer/zerosFill,dense/bias/Initializer/zeros/shape_as_tensor"dense/bias/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/bias*
_output_shapes	
:


dense/bias
VariableV2*
dtype0*
shared_name *
shape:*
	container *
_class
loc:@dense/bias*
_output_shapes	
:
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes	
:
l
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes	
:

dense/MatMulMatMulflatten/Reshapedense/kernel/read*
T0*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
dropout/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes

::
[
dropout/cond/switch_tIdentitydropout/cond/Switch:1*
T0
*
_output_shapes
:
Y
dropout/cond/switch_fIdentitydropout/cond/Switch*
T0
*
_output_shapes
:
P
dropout/cond/pred_idIdentityis_training*
T0
*
_output_shapes
:
{
dropout/cond/dropout/keep_probConst^dropout/cond/switch_t*
valueB
 *   ?*
dtype0*
_output_shapes
: 
}
dropout/cond/dropout/ShapeShape#dropout/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
š
!dropout/cond/dropout/Shape/SwitchSwitchdense/BiasAdddropout/cond/pred_id*
T0* 
_class
loc:@dense/BiasAdd*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

'dropout/cond/dropout/random_uniform/minConst^dropout/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 

'dropout/cond/dropout/random_uniform/maxConst^dropout/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ˇ
1dropout/cond/dropout/random_uniform/RandomUniformRandomUniformdropout/cond/dropout/Shape*
T0*
dtype0*
seed2 *

seed *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
'dropout/cond/dropout/random_uniform/subSub'dropout/cond/dropout/random_uniform/max'dropout/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
˝
'dropout/cond/dropout/random_uniform/mulMul1dropout/cond/dropout/random_uniform/RandomUniform'dropout/cond/dropout/random_uniform/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
#dropout/cond/dropout/random_uniformAdd'dropout/cond/dropout/random_uniform/mul'dropout/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout/cond/dropout/addAdddropout/cond/dropout/keep_prob#dropout/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
dropout/cond/dropout/FloorFloordropout/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout/cond/dropout/divRealDiv#dropout/cond/dropout/Shape/Switch:1dropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout/cond/dropout/mulMuldropout/cond/dropout/divdropout/cond/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
dropout/cond/IdentityIdentitydropout/cond/Identity/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
dropout/cond/Identity/SwitchSwitchdense/BiasAdddropout/cond/pred_id*
T0* 
_class
loc:@dense/BiasAdd*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout/cond/MergeMergedropout/cond/Identitydropout/cond/dropout/mul*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Ľ
1dense_1/kernel/Initializer/truncated_normal/shapeConst*
valueB"   
   *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:

0dense_1/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 

2dense_1/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *ĘM=*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
ô
;dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal1dense_1/kernel/Initializer/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed *!
_class
loc:@dense_1/kernel*
_output_shapes
:	

ô
/dense_1/kernel/Initializer/truncated_normal/mulMul;dense_1/kernel/Initializer/truncated_normal/TruncatedNormal2dense_1/kernel/Initializer/truncated_normal/stddev*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

â
+dense_1/kernel/Initializer/truncated_normalAdd/dense_1/kernel/Initializer/truncated_normal/mul0dense_1/kernel/Initializer/truncated_normal/mean*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

§
dense_1/kernel
VariableV2*
dtype0*
shared_name *
shape:	
*
	container *!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ň
dense_1/kernel/AssignAssigndense_1/kernel+dense_1/kernel/Initializer/truncated_normal*
T0*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	


dense_1/bias/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
:


dense_1/bias
VariableV2*
dtype0*
shared_name *
shape:
*
	container *
_class
loc:@dense_1/bias*
_output_shapes
:

ş
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:

q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:


dense_1/MatMulMatMuldropout/cond/Mergedense_1/kernel/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙


dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

U
scoresIdentitydense_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

h
&softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
v
'softmax_cross_entropy_with_logits/ShapeShapedense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
j
(softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
x
)softmax_cross_entropy_with_logits/Shape_1Shapedense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
i
'softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
 
%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 

-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*

axis *
T0*
N*
_output_shapes
:
v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ę
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:

1softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
o
-softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ů
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*
T0*
N*

Tidx0*
_output_shapes
:
¸
)softmax_cross_entropy_with_logits/ReshapeReshapedense_1/BiasAdd(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
j
(softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
j
)softmax_cross_entropy_with_logits/Shape_2Shapey*
T0*
out_type0*
_output_shapes
:
k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
¤
'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 

/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*

axis *
T0*
N*
_output_shapes
:
x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
đ
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:

3softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*
T0*
N*

Tidx0*
_output_shapes
:
Ž
+softmax_cross_entropy_with_logits/Reshape_1Reshapey*softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ä
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
˘
'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:

.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*

axis *
T0*
N*
_output_shapes
:
î
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ŕ
+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
MeanMean+softmax_cross_entropy_with_logits/Reshape_2Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

gradients/Mean_grad/ShapeShape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Mean_grad/Shape_1Shape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape!softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
č
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truediv@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/zeros_like	ZerosLike#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ż
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ł
4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*
T0*

Tdim0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
š
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
Ó
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ů
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:

@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ˇ
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

§
/gradients/dense_1/BiasAdd_grad/tuple/group_depsNoOp+^gradients/dense_1/BiasAdd_grad/BiasAddGradA^gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape
ž
7gradients/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape0^gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


9gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_1/BiasAdd_grad/BiasAddGrad0^gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

Ő
$gradients/dense_1/MatMul_grad/MatMulMatMul7gradients/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
&gradients/dense_1/MatMul_grad/MatMul_1MatMuldropout/cond/Merge7gradients/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	


.gradients/dense_1/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_1/MatMul_grad/MatMul'^gradients/dense_1/MatMul_grad/MatMul_1

6gradients/dense_1/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_1/MatMul_grad/MatMul/^gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

8gradients/dense_1/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_1/MatMul_grad/MatMul_1/^gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes
:	


+gradients/dropout/cond/Merge_grad/cond_gradSwitch6gradients/dense_1/MatMul_grad/tuple/control_dependencydropout/cond/pred_id*
T0*7
_class-
+)loc:@gradients/dense_1/MatMul_grad/MatMul*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
h
2gradients/dropout/cond/Merge_grad/tuple/group_depsNoOp,^gradients/dropout/cond/Merge_grad/cond_grad

:gradients/dropout/cond/Merge_grad/tuple/control_dependencyIdentity+gradients/dropout/cond/Merge_grad/cond_grad3^gradients/dropout/cond/Merge_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients/dropout/cond/Merge_grad/tuple/control_dependency_1Identity-gradients/dropout/cond/Merge_grad/cond_grad:13^gradients/dropout/cond/Merge_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients/dropout/cond/dropout/mul_grad/ShapeShapedropout/cond/dropout/div*
T0*
out_type0*
_output_shapes
:

/gradients/dropout/cond/dropout/mul_grad/Shape_1Shapedropout/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
ó
=gradients/dropout/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/dropout/cond/dropout/mul_grad/Shape/gradients/dropout/cond/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ż
+gradients/dropout/cond/dropout/mul_grad/MulMul<gradients/dropout/cond/Merge_grad/tuple/control_dependency_1dropout/cond/dropout/Floor*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
+gradients/dropout/cond/dropout/mul_grad/SumSum+gradients/dropout/cond/dropout/mul_grad/Mul=gradients/dropout/cond/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
×
/gradients/dropout/cond/dropout/mul_grad/ReshapeReshape+gradients/dropout/cond/dropout/mul_grad/Sum-gradients/dropout/cond/dropout/mul_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
-gradients/dropout/cond/dropout/mul_grad/Mul_1Muldropout/cond/dropout/div<gradients/dropout/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
-gradients/dropout/cond/dropout/mul_grad/Sum_1Sum-gradients/dropout/cond/dropout/mul_grad/Mul_1?gradients/dropout/cond/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ý
1gradients/dropout/cond/dropout/mul_grad/Reshape_1Reshape-gradients/dropout/cond/dropout/mul_grad/Sum_1/gradients/dropout/cond/dropout/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
8gradients/dropout/cond/dropout/mul_grad/tuple/group_depsNoOp0^gradients/dropout/cond/dropout/mul_grad/Reshape2^gradients/dropout/cond/dropout/mul_grad/Reshape_1
Ż
@gradients/dropout/cond/dropout/mul_grad/tuple/control_dependencyIdentity/gradients/dropout/cond/dropout/mul_grad/Reshape9^gradients/dropout/cond/dropout/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/dropout/cond/dropout/mul_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
Bgradients/dropout/cond/dropout/mul_grad/tuple/control_dependency_1Identity1gradients/dropout/cond/dropout/mul_grad/Reshape_19^gradients/dropout/cond/dropout/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/dropout/cond/dropout/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/SwitchSwitchdense/BiasAdddropout/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
e
gradients/IdentityIdentitygradients/Switch:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
o
gradients/zeros/ConstConst^gradients/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
5gradients/dropout/cond/Identity/Switch_grad/cond_gradMerge:gradients/dropout/cond/Merge_grad/tuple/control_dependencygradients/zeros*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 

-gradients/dropout/cond/dropout/div_grad/ShapeShape#dropout/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
r
/gradients/dropout/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ó
=gradients/dropout/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/dropout/cond/dropout/div_grad/Shape/gradients/dropout/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ď
/gradients/dropout/cond/dropout/div_grad/RealDivRealDiv@gradients/dropout/cond/dropout/mul_grad/tuple/control_dependencydropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
+gradients/dropout/cond/dropout/div_grad/SumSum/gradients/dropout/cond/dropout/div_grad/RealDiv=gradients/dropout/cond/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
×
/gradients/dropout/cond/dropout/div_grad/ReshapeReshape+gradients/dropout/cond/dropout/div_grad/Sum-gradients/dropout/cond/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients/dropout/cond/dropout/div_grad/NegNeg#dropout/cond/dropout/Shape/Switch:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
1gradients/dropout/cond/dropout/div_grad/RealDiv_1RealDiv+gradients/dropout/cond/dropout/div_grad/Negdropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
1gradients/dropout/cond/dropout/div_grad/RealDiv_2RealDiv1gradients/dropout/cond/dropout/div_grad/RealDiv_1dropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ú
+gradients/dropout/cond/dropout/div_grad/mulMul@gradients/dropout/cond/dropout/mul_grad/tuple/control_dependency1gradients/dropout/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
-gradients/dropout/cond/dropout/div_grad/Sum_1Sum+gradients/dropout/cond/dropout/div_grad/mul?gradients/dropout/cond/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ë
1gradients/dropout/cond/dropout/div_grad/Reshape_1Reshape-gradients/dropout/cond/dropout/div_grad/Sum_1/gradients/dropout/cond/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ś
8gradients/dropout/cond/dropout/div_grad/tuple/group_depsNoOp0^gradients/dropout/cond/dropout/div_grad/Reshape2^gradients/dropout/cond/dropout/div_grad/Reshape_1
Ż
@gradients/dropout/cond/dropout/div_grad/tuple/control_dependencyIdentity/gradients/dropout/cond/dropout/div_grad/Reshape9^gradients/dropout/cond/dropout/div_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/dropout/cond/dropout/div_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients/dropout/cond/dropout/div_grad/tuple/control_dependency_1Identity1gradients/dropout/cond/dropout/div_grad/Reshape_19^gradients/dropout/cond/dropout/div_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/dropout/cond/dropout/div_grad/Reshape_1*
_output_shapes
: 

gradients/Switch_1Switchdense/BiasAdddropout/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
g
gradients/Identity_1Identitygradients/Switch_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
gradients/Shape_2Shapegradients/Switch_1*
T0*
out_type0*
_output_shapes
:
s
gradients/zeros_1/ConstConst^gradients/Identity_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ö
:gradients/dropout/cond/dropout/Shape/Switch_grad/cond_gradMergegradients/zeros_1@gradients/dropout/cond/dropout/div_grad/tuple/control_dependency*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 

gradients/AddNAddN5gradients/dropout/cond/Identity/Switch_grad/cond_grad:gradients/dropout/cond/dropout/Shape/Switch_grad/cond_grad*
T0*
N*H
_class>
<:loc:@gradients/dropout/cond/Identity/Switch_grad/cond_grad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes	
:
q
-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN)^gradients/dense/BiasAdd_grad/BiasAddGrad
ţ
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/dropout/cond/Identity/Switch_grad/cond_grad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ď
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ç
$gradients/dense/MatMul_grad/MatMul_1MatMulflatten/Reshape5gradients/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:
 

,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
ý
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ű
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
 
{
$gradients/flatten/Reshape_grad/ShapeShapemax_pooling2d_1/MaxPool*
T0*
out_type0*
_output_shapes
:
Ő
&gradients/flatten/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependency$gradients/flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

2gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_1/Relumax_pooling2d_1/MaxPool&gradients/flatten/Reshape_grad/Reshape*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ž
%gradients/conv2d_1/Relu_grad/ReluGradReluGrad2gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradconv2d_1/Relu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

+gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/conv2d_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

0gradients/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp,^gradients/conv2d_1/BiasAdd_grad/BiasAddGrad&^gradients/conv2d_1/Relu_grad/ReluGrad

8gradients/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/conv2d_1/Relu_grad/ReluGrad1^gradients/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/conv2d_1/Relu_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

:gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/conv2d_1/BiasAdd_grad/BiasAddGrad1^gradients/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
 
%gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
č
2gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/kernel/read8gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ä
3gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool'gradients/conv2d_1/Conv2D_grad/ShapeN:18gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
:  
˘
/gradients/conv2d_1/Conv2D_grad/tuple/group_depsNoOp4^gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter3^gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput
Ş
7gradients/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ľ
9gradients/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:  
Š
0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPool7gradients/conv2d_1/Conv2D_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize
*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
¨
#gradients/conv2d/Relu_grad/ReluGradReluGrad0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

)gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/conv2d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 

.gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp*^gradients/conv2d/BiasAdd_grad/BiasAddGrad$^gradients/conv2d/Relu_grad/ReluGrad

6gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/conv2d/Relu_grad/ReluGrad/^gradients/conv2d/BiasAdd_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/conv2d/Relu_grad/ReluGrad*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

8gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity)gradients/conv2d/BiasAdd_grad/BiasAddGrad/^gradients/conv2d/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/conv2d/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

#gradients/conv2d/Conv2D_grad/ShapeNShapeNReshapeconv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
ŕ
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/read6gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshape%gradients/conv2d/Conv2D_grad/ShapeN:16gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(*&
_output_shapes
: 

-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter1^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput
˘
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1Identity1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
~
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 

beta1_power
VariableV2*
dtype0*
shared_name *
shape: *
	container *
_class
loc:@conv2d/bias*
_output_shapes
: 
Ž
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
j
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 
~
beta2_power/initial_valueConst*
valueB
 *wž?*
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 

beta2_power
VariableV2*
dtype0*
shared_name *
shape: *
	container *
_class
loc:@conv2d/bias*
_output_shapes
: 
Ž
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
j
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 
Ť
$conv2d/kernel/Adam/Initializer/zerosConst*%
valueB *    *
dtype0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
¸
conv2d/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape: *
	container * 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
Ů
conv2d/kernel/Adam/AssignAssignconv2d/kernel/Adam$conv2d/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 

conv2d/kernel/Adam/readIdentityconv2d/kernel/Adam*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
­
&conv2d/kernel/Adam_1/Initializer/zerosConst*%
valueB *    *
dtype0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
ş
conv2d/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape: *
	container * 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
ß
conv2d/kernel/Adam_1/AssignAssignconv2d/kernel/Adam_1&conv2d/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 

conv2d/kernel/Adam_1/readIdentityconv2d/kernel/Adam_1*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 

"conv2d/bias/Adam/Initializer/zerosConst*
valueB *    *
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 

conv2d/bias/Adam
VariableV2*
dtype0*
shared_name *
shape: *
	container *
_class
loc:@conv2d/bias*
_output_shapes
: 
Ĺ
conv2d/bias/Adam/AssignAssignconv2d/bias/Adam"conv2d/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
x
conv2d/bias/Adam/readIdentityconv2d/bias/Adam*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 

$conv2d/bias/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*
_class
loc:@conv2d/bias*
_output_shapes
: 

conv2d/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape: *
	container *
_class
loc:@conv2d/bias*
_output_shapes
: 
Ë
conv2d/bias/Adam_1/AssignAssignconv2d/bias/Adam_1$conv2d/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
|
conv2d/bias/Adam_1/readIdentityconv2d/bias/Adam_1*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 
ł
6conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"              *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:

,conv2d_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
ű
&conv2d_1/kernel/Adam/Initializer/zerosFill6conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensor,conv2d_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ź
conv2d_1/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:  *
	container *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
á
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adam&conv2d_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  

conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ľ
8conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"              *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
:

.conv2d_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 

(conv2d_1/kernel/Adam_1/Initializer/zerosFill8conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor.conv2d_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ž
conv2d_1/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:  *
	container *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ç
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1(conv2d_1/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  

conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  

$conv2d_1/bias/Adam/Initializer/zerosConst*
valueB *    *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
 
conv2d_1/bias/Adam
VariableV2*
dtype0*
shared_name *
shape: *
	container * 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
Í
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adam$conv2d_1/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 

&conv2d_1/bias/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
˘
conv2d_1/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape: *
	container * 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
Ó
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1&conv2d_1/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 

conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
Ľ
3dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:

)dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
é
#dense/kernel/Adam/Initializer/zerosFill3dense/kernel/Adam/Initializer/zeros/shape_as_tensor)dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/kernel* 
_output_shapes
:
 
Ş
dense/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:
 *
	container *
_class
loc:@dense/kernel* 
_output_shapes
:
 
Ď
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel* 
_output_shapes
:
 

dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
 
§
5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:

+dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 
ď
%dense/kernel/Adam_1/Initializer/zerosFill5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor+dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/kernel* 
_output_shapes
:
 
Ź
dense/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:
 *
	container *
_class
loc:@dense/kernel* 
_output_shapes
:
 
Ő
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel* 
_output_shapes
:
 

dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
 

1dense/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:*
dtype0*
_class
loc:@dense/bias*
_output_shapes
:

'dense/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_class
loc:@dense/bias*
_output_shapes
: 
Ü
!dense/bias/Adam/Initializer/zerosFill1dense/bias/Adam/Initializer/zeros/shape_as_tensor'dense/bias/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/bias*
_output_shapes	
:

dense/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:*
	container *
_class
loc:@dense/bias*
_output_shapes	
:
Â
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes	
:
v
dense/bias/Adam/readIdentitydense/bias/Adam*
T0*
_class
loc:@dense/bias*
_output_shapes	
:

3dense/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:*
dtype0*
_class
loc:@dense/bias*
_output_shapes
:

)dense/bias/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_class
loc:@dense/bias*
_output_shapes
: 
â
#dense/bias/Adam_1/Initializer/zerosFill3dense/bias/Adam_1/Initializer/zeros/shape_as_tensor)dense/bias/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/bias*
_output_shapes	
:

dense/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:*
	container *
_class
loc:@dense/bias*
_output_shapes	
:
Č
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes	
:
z
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias*
_output_shapes	
:
Š
5dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:

+dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
đ
%dense_1/kernel/Adam/Initializer/zerosFill5dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor+dense_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ź
dense_1/kernel/Adam
VariableV2*
dtype0*
shared_name *
shape:	
*
	container *!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ö
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adam%dense_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	


dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ť
7dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
:

-dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
ö
'dense_1/kernel/Adam_1/Initializer/zerosFill7dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor-dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ž
dense_1/kernel/Adam_1
VariableV2*
dtype0*
shared_name *
shape:	
*
	container *!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Ü
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1'dense_1/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	


dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	


#dense_1/bias/Adam/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
:


dense_1/bias/Adam
VariableV2*
dtype0*
shared_name *
shape:
*
	container *
_class
loc:@dense_1/bias*
_output_shapes
:

É
dense_1/bias/Adam/AssignAssigndense_1/bias/Adam#dense_1/bias/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:

{
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:


%dense_1/bias/Adam_1/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
:

 
dense_1/bias/Adam_1
VariableV2*
dtype0*
shared_name *
shape:
*
	container *
_class
loc:@dense_1/bias*
_output_shapes
:

Ď
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1%dense_1/bias/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:


dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:

W
Adam/learning_rateConst*
valueB
 *ˇŃ8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ú
#Adam/update_conv2d/kernel/ApplyAdam	ApplyAdamconv2d/kernelconv2d/kernel/Adamconv2d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( * 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
ĺ
!Adam/update_conv2d/bias/ApplyAdam	ApplyAdamconv2d/biasconv2d/bias/Adamconv2d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@conv2d/bias*
_output_shapes
: 

%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv2d_1/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ń
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( * 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
î
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@dense/kernel* 
_output_shapes
:
 
ŕ
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@dense/bias*
_output_shapes	
:
ů
$Adam/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *!
_class
loc:@dense_1/kernel*
_output_shapes
:	

ë
"Adam/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@dense_1/bias*
_output_shapes
:


Adam/mulMulbeta1_power/read
Adam/beta1"^Adam/update_conv2d/bias/ApplyAdam$^Adam/update_conv2d/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_1/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
use_locking( *
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2"^Adam/update_conv2d/bias/ApplyAdam$^Adam/update_conv2d/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_1/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam*
T0*
_class
loc:@conv2d/bias*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
use_locking( *
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
Ö
AdamNoOp^Adam/Assign^Adam/Assign_1"^Adam/update_conv2d/bias/ApplyAdam$^Adam/update_conv2d/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam&^Adam/update_conv2d_1/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam#^Adam/update_dense_1/bias/ApplyAdam%^Adam/update_dense_1/kernel/ApplyAdam
˛
initNoOp^beta1_power/Assign^beta2_power/Assign^conv2d/bias/Adam/Assign^conv2d/bias/Adam_1/Assign^conv2d/bias/Assign^conv2d/kernel/Adam/Assign^conv2d/kernel/Adam_1/Assign^conv2d/kernel/Assign^conv2d_1/bias/Adam/Assign^conv2d_1/bias/Adam_1/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Adam/Assign^conv2d_1/kernel/Adam_1/Assign^conv2d_1/kernel/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^dense_1/bias/Adam/Assign^dense_1/bias/Adam_1/Assign^dense_1/bias/Assign^dense_1/kernel/Adam/Assign^dense_1/kernel/Adam_1/Assign^dense_1/kernel/Assign
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
r
ArgMaxArgMaxyArgMax/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_1ArgMaxdense_1/BiasAddArgMax_1/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
`
CastCastEqual*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
W
SumSumCastConst_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
T
ArgMax_2/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
v
ArgMax_2ArgMaxyArgMax_2/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_3/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_3ArgMaxdense_1/BiasAddArgMax_3/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Equal_1EqualArgMax_2ArgMax_3*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_1CastEqual_1*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_1SumCast_1Const_2*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
T
ArgMax_4/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
v
ArgMax_4ArgMaxyArgMax_4/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_5/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_5ArgMaxdense_1/BiasAddArgMax_5/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Equal_2EqualArgMax_4ArgMax_5*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_2CastEqual_2*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_3Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_2SumCast_2Const_3*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
T
ArgMax_6/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
v
ArgMax_6ArgMaxyArgMax_6/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_7/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_7ArgMaxdense_1/BiasAddArgMax_7/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Equal_3EqualArgMax_6ArgMax_7*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_3CastEqual_3*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_3SumCast_3Const_4*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
T
ArgMax_8/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
v
ArgMax_8ArgMaxyArgMax_8/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
ArgMax_9/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_9ArgMaxdense_1/BiasAddArgMax_9/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Equal_4EqualArgMax_8ArgMax_9*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_4CastEqual_4*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_5Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_4SumCast_4Const_5*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_10/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_10ArgMaxyArgMax_10/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_11/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_11ArgMaxdense_1/BiasAddArgMax_11/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
Equal_5Equal	ArgMax_10	ArgMax_11*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_5CastEqual_5*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_6Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_5SumCast_5Const_6*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_12/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_12ArgMaxyArgMax_12/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_13/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_13ArgMaxdense_1/BiasAddArgMax_13/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
Equal_6Equal	ArgMax_12	ArgMax_13*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_6CastEqual_6*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_7Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_6SumCast_6Const_7*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_14/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_14ArgMaxyArgMax_14/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_15/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_15ArgMaxdense_1/BiasAddArgMax_15/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
Equal_7Equal	ArgMax_14	ArgMax_15*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_7CastEqual_7*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_8Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_7SumCast_7Const_8*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_16/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_16ArgMaxyArgMax_16/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_17/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_17ArgMaxdense_1/BiasAddArgMax_17/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
Equal_8Equal	ArgMax_16	ArgMax_17*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_8CastEqual_8*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_9Const*
valueB: *
dtype0*
_output_shapes
:
[
Sum_8SumCast_8Const_9*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_18/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_18ArgMaxyArgMax_18/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_19/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_19ArgMaxdense_1/BiasAddArgMax_19/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
Equal_9Equal	ArgMax_18	ArgMax_19*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
Cast_9CastEqual_9*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Const_10Const*
valueB: *
dtype0*
_output_shapes
:
\
Sum_9SumCast_9Const_10*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_20/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_20ArgMaxyArgMax_20/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_21/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_21ArgMaxdense_1/BiasAddArgMax_21/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
Equal_10Equal	ArgMax_20	ArgMax_21*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
Cast_10CastEqual_10*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Const_11Const*
valueB: *
dtype0*
_output_shapes
:
^
Sum_10SumCast_10Const_11*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_22/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_22ArgMaxyArgMax_22/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_23/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_23ArgMaxdense_1/BiasAddArgMax_23/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
Equal_11Equal	ArgMax_22	ArgMax_23*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
Cast_11CastEqual_11*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Const_12Const*
valueB: *
dtype0*
_output_shapes
:
^
Sum_11SumCast_11Const_12*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_24/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_24ArgMaxyArgMax_24/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_25/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_25ArgMaxdense_1/BiasAddArgMax_25/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
Equal_12Equal	ArgMax_24	ArgMax_25*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
Cast_12CastEqual_12*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Const_13Const*
valueB: *
dtype0*
_output_shapes
:
^
Sum_12SumCast_12Const_13*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_26/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_26ArgMaxyArgMax_26/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_27/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_27ArgMaxdense_1/BiasAddArgMax_27/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
Equal_13Equal	ArgMax_26	ArgMax_27*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
Cast_13CastEqual_13*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Const_14Const*
valueB: *
dtype0*
_output_shapes
:
^
Sum_13SumCast_13Const_14*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
U
ArgMax_28/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
	ArgMax_28ArgMaxyArgMax_28/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ArgMax_29/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

	ArgMax_29ArgMaxdense_1/BiasAddArgMax_29/dimension*
output_type0	*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
Equal_14Equal	ArgMax_28	ArgMax_29*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
Cast_14CastEqual_14*

DstT0*
Truncate( *

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
R
Const_15Const*
valueB: *
dtype0*
_output_shapes
:
^
Sum_14SumCast_14Const_15*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_438fd71685a84eb4938d944e6822eabc/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
¸
save/SaveV2/tensor_namesConst*ë
valueáBŢBbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ă
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerconv2d/biasconv2d/bias/Adamconv2d/bias/Adam_1conv2d/kernelconv2d/kernel/Adamconv2d/kernel/Adam_1conv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1conv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1*(
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
ť
save/RestoreV2/tensor_namesConst*ë
valueáBŢBbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*(
dtypes
2*|
_output_shapesj
h::::::::::::::::::::::::::

save/AssignAssignbeta1_powersave/RestoreV2*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
 
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
¤
save/Assign_2Assignconv2d/biassave/RestoreV2:2*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
Š
save/Assign_3Assignconv2d/bias/Adamsave/RestoreV2:3*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
Ť
save/Assign_4Assignconv2d/bias/Adam_1save/RestoreV2:4*
T0*
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias*
_output_shapes
: 
´
save/Assign_5Assignconv2d/kernelsave/RestoreV2:5*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
š
save/Assign_6Assignconv2d/kernel/Adamsave/RestoreV2:6*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
ť
save/Assign_7Assignconv2d/kernel/Adam_1save/RestoreV2:7*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel*&
_output_shapes
: 
¨
save/Assign_8Assignconv2d_1/biassave/RestoreV2:8*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
­
save/Assign_9Assignconv2d_1/bias/Adamsave/RestoreV2:9*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
ą
save/Assign_10Assignconv2d_1/bias/Adam_1save/RestoreV2:10*
T0*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias*
_output_shapes
: 
ş
save/Assign_11Assignconv2d_1/kernelsave/RestoreV2:11*
T0*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
ż
save/Assign_12Assignconv2d_1/kernel/Adamsave/RestoreV2:12*
T0*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
Á
save/Assign_13Assignconv2d_1/kernel/Adam_1save/RestoreV2:13*
T0*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:  
Ľ
save/Assign_14Assign
dense/biassave/RestoreV2:14*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes	
:
Ş
save/Assign_15Assigndense/bias/Adamsave/RestoreV2:15*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes	
:
Ź
save/Assign_16Assigndense/bias/Adam_1save/RestoreV2:16*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/bias*
_output_shapes	
:
Ž
save/Assign_17Assigndense/kernelsave/RestoreV2:17*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel* 
_output_shapes
:
 
ł
save/Assign_18Assigndense/kernel/Adamsave/RestoreV2:18*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel* 
_output_shapes
:
 
ľ
save/Assign_19Assigndense/kernel/Adam_1save/RestoreV2:19*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel* 
_output_shapes
:
 
¨
save/Assign_20Assigndense_1/biassave/RestoreV2:20*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:

­
save/Assign_21Assigndense_1/bias/Adamsave/RestoreV2:21*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:

Ż
save/Assign_22Assigndense_1/bias/Adam_1save/RestoreV2:22*
T0*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:

ą
save/Assign_23Assigndense_1/kernelsave/RestoreV2:23*
T0*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

ś
save/Assign_24Assigndense_1/kernel/Adamsave/RestoreV2:24*
T0*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

¸
save/Assign_25Assigndense_1/kernel/Adam_1save/RestoreV2:25*
T0*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes
:	

Č
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"×
trainable_variablesżź
m
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02,conv2d/kernel/Initializer/truncated_normal:08
Z
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:08
u
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02.conv2d_1/kernel/Initializer/truncated_normal:08
b
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:08
i
dense/kernel:0dense/kernel/Assigndense/kernel/read:02+dense/kernel/Initializer/truncated_normal:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
q
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02-dense_1/kernel/Initializer/truncated_normal:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08"ű
	variablesíę
m
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02,conv2d/kernel/Initializer/truncated_normal:08
Z
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:08
u
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02.conv2d_1/kernel/Initializer/truncated_normal:08
b
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:08
i
dense/kernel:0dense/kernel/Assigndense/kernel/read:02+dense/kernel/Initializer/truncated_normal:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08
q
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02-dense_1/kernel/Initializer/truncated_normal:08
^
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
t
conv2d/kernel/Adam:0conv2d/kernel/Adam/Assignconv2d/kernel/Adam/read:02&conv2d/kernel/Adam/Initializer/zeros:0
|
conv2d/kernel/Adam_1:0conv2d/kernel/Adam_1/Assignconv2d/kernel/Adam_1/read:02(conv2d/kernel/Adam_1/Initializer/zeros:0
l
conv2d/bias/Adam:0conv2d/bias/Adam/Assignconv2d/bias/Adam/read:02$conv2d/bias/Adam/Initializer/zeros:0
t
conv2d/bias/Adam_1:0conv2d/bias/Adam_1/Assignconv2d/bias/Adam_1/read:02&conv2d/bias/Adam_1/Initializer/zeros:0
|
conv2d_1/kernel/Adam:0conv2d_1/kernel/Adam/Assignconv2d_1/kernel/Adam/read:02(conv2d_1/kernel/Adam/Initializer/zeros:0

conv2d_1/kernel/Adam_1:0conv2d_1/kernel/Adam_1/Assignconv2d_1/kernel/Adam_1/read:02*conv2d_1/kernel/Adam_1/Initializer/zeros:0
t
conv2d_1/bias/Adam:0conv2d_1/bias/Adam/Assignconv2d_1/bias/Adam/read:02&conv2d_1/bias/Adam/Initializer/zeros:0
|
conv2d_1/bias/Adam_1:0conv2d_1/bias/Adam_1/Assignconv2d_1/bias/Adam_1/read:02(conv2d_1/bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0
x
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assigndense_1/kernel/Adam/read:02'dense_1/kernel/Adam/Initializer/zeros:0

dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assigndense_1/kernel/Adam_1/read:02)dense_1/kernel/Adam_1/Initializer/zeros:0
p
dense_1/bias/Adam:0dense_1/bias/Adam/Assigndense_1/bias/Adam/read:02%dense_1/bias/Adam/Initializer/zeros:0
x
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assigndense_1/bias/Adam_1/read:02'dense_1/bias/Adam_1/Initializer/zeros:0"Ą
cond_context
Ů
dropout/cond/cond_textdropout/cond/pred_id:0dropout/cond/switch_t:0 *
dense/BiasAdd:0
dropout/cond/dropout/Floor:0
#dropout/cond/dropout/Shape/Switch:1
dropout/cond/dropout/Shape:0
dropout/cond/dropout/add:0
dropout/cond/dropout/div:0
 dropout/cond/dropout/keep_prob:0
dropout/cond/dropout/mul:0
3dropout/cond/dropout/random_uniform/RandomUniform:0
)dropout/cond/dropout/random_uniform/max:0
)dropout/cond/dropout/random_uniform/min:0
)dropout/cond/dropout/random_uniform/mul:0
)dropout/cond/dropout/random_uniform/sub:0
%dropout/cond/dropout/random_uniform:0
dropout/cond/pred_id:0
dropout/cond/switch_t:00
dropout/cond/pred_id:0dropout/cond/pred_id:06
dense/BiasAdd:0#dropout/cond/dropout/Shape/Switch:1
Ž
dropout/cond/cond_text_1dropout/cond/pred_id:0dropout/cond/switch_f:0*ŕ
dense/BiasAdd:0
dropout/cond/Identity/Switch:0
dropout/cond/Identity:0
dropout/cond/pred_id:0
dropout/cond/switch_f:00
dropout/cond/pred_id:0dropout/cond/pred_id:01
dense/BiasAdd:0dropout/cond/Identity/Switch:0"
train_op

Adam*Ź
serving_default
 
X
X:0˙˙˙˙˙˙˙˙˙
$
is_training
is_training:0
2
scores(
dense_1/BiasAdd:0˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict