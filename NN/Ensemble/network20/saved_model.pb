��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02unknown8��
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
z
dense_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_96/kernel
s
#dense_96/kernel/Read/ReadVariableOpReadVariableOpdense_96/kernel*
_output_shapes

:
*
dtype0
r
dense_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_96/bias
k
!dense_96/bias/Read/ReadVariableOpReadVariableOpdense_96/bias*
_output_shapes
:
*
dtype0
z
dense_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_97/kernel
s
#dense_97/kernel/Read/ReadVariableOpReadVariableOpdense_97/kernel*
_output_shapes

:

*
dtype0
r
dense_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_97/bias
k
!dense_97/bias/Read/ReadVariableOpReadVariableOpdense_97/bias*
_output_shapes
:
*
dtype0
z
dense_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_98/kernel
s
#dense_98/kernel/Read/ReadVariableOpReadVariableOpdense_98/kernel*
_output_shapes

:

*
dtype0
r
dense_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_98/bias
k
!dense_98/bias/Read/ReadVariableOpReadVariableOpdense_98/bias*
_output_shapes
:
*
dtype0
z
dense_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_99/kernel
s
#dense_99/kernel/Read/ReadVariableOpReadVariableOpdense_99/kernel*
_output_shapes

:
*
dtype0
r
dense_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_99/bias
k
!dense_99/bias/Read/ReadVariableOpReadVariableOpdense_99/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Nadam/dense_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_96/kernel/m
�
+Nadam/dense_96/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_96/kernel/m*
_output_shapes

:
*
dtype0
�
Nadam/dense_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_96/bias/m
{
)Nadam/dense_96/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_96/bias/m*
_output_shapes
:
*
dtype0
�
Nadam/dense_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameNadam/dense_97/kernel/m
�
+Nadam/dense_97/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_97/kernel/m*
_output_shapes

:

*
dtype0
�
Nadam/dense_97/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_97/bias/m
{
)Nadam/dense_97/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_97/bias/m*
_output_shapes
:
*
dtype0
�
Nadam/dense_98/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameNadam/dense_98/kernel/m
�
+Nadam/dense_98/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_98/kernel/m*
_output_shapes

:

*
dtype0
�
Nadam/dense_98/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_98/bias/m
{
)Nadam/dense_98/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_98/bias/m*
_output_shapes
:
*
dtype0
�
Nadam/dense_99/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_99/kernel/m
�
+Nadam/dense_99/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_99/kernel/m*
_output_shapes

:
*
dtype0
�
Nadam/dense_99/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_99/bias/m
{
)Nadam/dense_99/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_99/bias/m*
_output_shapes
:*
dtype0
�
Nadam/dense_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_96/kernel/v
�
+Nadam/dense_96/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_96/kernel/v*
_output_shapes

:
*
dtype0
�
Nadam/dense_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_96/bias/v
{
)Nadam/dense_96/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_96/bias/v*
_output_shapes
:
*
dtype0
�
Nadam/dense_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameNadam/dense_97/kernel/v
�
+Nadam/dense_97/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_97/kernel/v*
_output_shapes

:

*
dtype0
�
Nadam/dense_97/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_97/bias/v
{
)Nadam/dense_97/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_97/bias/v*
_output_shapes
:
*
dtype0
�
Nadam/dense_98/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*(
shared_nameNadam/dense_98/kernel/v
�
+Nadam/dense_98/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_98/kernel/v*
_output_shapes

:

*
dtype0
�
Nadam/dense_98/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameNadam/dense_98/bias/v
{
)Nadam/dense_98/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_98/bias/v*
_output_shapes
:
*
dtype0
�
Nadam/dense_99/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*(
shared_nameNadam/dense_99/kernel/v
�
+Nadam/dense_99/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_99/kernel/v*
_output_shapes

:
*
dtype0
�
Nadam/dense_99/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_99/bias/v
{
)Nadam/dense_99/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_99/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�0
value�/B�/ B�/
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
]
state_variables
_broadcast_shape
mean
variance
	count
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
�
*iter

+beta_1

,beta_2
	-decay
.learning_rate
/momentum_cachemNmOmPmQmRmS$mT%mUvVvWvXvYvZv[$v\%v]
 
8
0
1
2
3
4
5
$6
%7
N
0
1
2
3
4
5
6
7
8
$9
%10
�
0non_trainable_variables
1layer_metrics
regularization_losses
2layer_regularization_losses
3metrics

4layers
trainable_variables
		variables
 
#
mean
variance
	count
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 
[Y
VARIABLE_VALUEdense_96/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_96/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
5non_trainable_variables
6layer_metrics
regularization_losses
7metrics
8layer_regularization_losses

9layers
	variables
trainable_variables
[Y
VARIABLE_VALUEdense_97/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_97/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
:non_trainable_variables
;layer_metrics
regularization_losses
<metrics
=layer_regularization_losses

>layers
	variables
trainable_variables
[Y
VARIABLE_VALUEdense_98/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_98/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
?non_trainable_variables
@layer_metrics
 regularization_losses
Ametrics
Blayer_regularization_losses

Clayers
!	variables
"trainable_variables
[Y
VARIABLE_VALUEdense_99/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_99/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
�
Dnon_trainable_variables
Elayer_metrics
&regularization_losses
Fmetrics
Glayer_regularization_losses

Hlayers
'	variables
(trainable_variables
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 

I0
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Jtotal
	Kcount
L	variables
M	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

L	variables
}
VARIABLE_VALUENadam/dense_96/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_96/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_97/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_97/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_98/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_98/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_99/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_99/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_96/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_96/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_97/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_97/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_98/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_98/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_99/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_99/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
&serving_default_normalization_24_inputPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_24_inputmeanvariancedense_96/kerneldense_96/biasdense_97/kerneldense_97/biasdense_98/kerneldense_98/biasdense_99/kerneldense_99/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6124199
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp#dense_96/kernel/Read/ReadVariableOp!dense_96/bias/Read/ReadVariableOp#dense_97/kernel/Read/ReadVariableOp!dense_97/bias/Read/ReadVariableOp#dense_98/kernel/Read/ReadVariableOp!dense_98/bias/Read/ReadVariableOp#dense_99/kernel/Read/ReadVariableOp!dense_99/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Nadam/dense_96/kernel/m/Read/ReadVariableOp)Nadam/dense_96/bias/m/Read/ReadVariableOp+Nadam/dense_97/kernel/m/Read/ReadVariableOp)Nadam/dense_97/bias/m/Read/ReadVariableOp+Nadam/dense_98/kernel/m/Read/ReadVariableOp)Nadam/dense_98/bias/m/Read/ReadVariableOp+Nadam/dense_99/kernel/m/Read/ReadVariableOp)Nadam/dense_99/bias/m/Read/ReadVariableOp+Nadam/dense_96/kernel/v/Read/ReadVariableOp)Nadam/dense_96/bias/v/Read/ReadVariableOp+Nadam/dense_97/kernel/v/Read/ReadVariableOp)Nadam/dense_97/bias/v/Read/ReadVariableOp+Nadam/dense_98/kernel/v/Read/ReadVariableOp)Nadam/dense_98/bias/v/Read/ReadVariableOp+Nadam/dense_99/kernel/v/Read/ReadVariableOp)Nadam/dense_99/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_6124540
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_96/kerneldense_96/biasdense_97/kerneldense_97/biasdense_98/kerneldense_98/biasdense_99/kerneldense_99/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcount_1Nadam/dense_96/kernel/mNadam/dense_96/bias/mNadam/dense_97/kernel/mNadam/dense_97/bias/mNadam/dense_98/kernel/mNadam/dense_98/bias/mNadam/dense_99/kernel/mNadam/dense_99/bias/mNadam/dense_96/kernel/vNadam/dense_96/bias/vNadam/dense_97/kernel/vNadam/dense_97/bias/vNadam/dense_98/kernel/vNadam/dense_98/bias/vNadam/dense_99/kernel/vNadam/dense_99/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_6124655��
�
�
%__inference_signature_wrapper_6124199
normalization_24_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_61238852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:������������������
0
_user_specified_namenormalization_24_input
�	
�
/__inference_sequential_24_layer_call_fn_6124104
normalization_24_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_61240812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:������������������
0
_user_specified_namenormalization_24_input
�*
�
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124283

inputs4
0normalization_24_reshape_readvariableop_resource6
2normalization_24_reshape_1_readvariableop_resource+
'dense_96_matmul_readvariableop_resource,
(dense_96_biasadd_readvariableop_resource+
'dense_97_matmul_readvariableop_resource,
(dense_97_biasadd_readvariableop_resource+
'dense_98_matmul_readvariableop_resource,
(dense_98_biasadd_readvariableop_resource+
'dense_99_matmul_readvariableop_resource,
(dense_99_biasadd_readvariableop_resource
identity��
'normalization_24/Reshape/ReadVariableOpReadVariableOp0normalization_24_reshape_readvariableop_resource*
_output_shapes
:*
dtype02)
'normalization_24/Reshape/ReadVariableOp�
normalization_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_24/Reshape/shape�
normalization_24/ReshapeReshape/normalization_24/Reshape/ReadVariableOp:value:0'normalization_24/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape�
)normalization_24/Reshape_1/ReadVariableOpReadVariableOp2normalization_24_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)normalization_24/Reshape_1/ReadVariableOp�
 normalization_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_24/Reshape_1/shape�
normalization_24/Reshape_1Reshape1normalization_24/Reshape_1/ReadVariableOp:value:0)normalization_24/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape_1�
normalization_24/subSubinputs!normalization_24/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_24/sub�
normalization_24/SqrtSqrt#normalization_24/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_24/Sqrt�
normalization_24/truedivRealDivnormalization_24/sub:z:0normalization_24/Sqrt:y:0*
T0*'
_output_shapes
:���������2
normalization_24/truediv�
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_96/MatMul/ReadVariableOp�
dense_96/MatMulMatMulnormalization_24/truediv:z:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_96/MatMul�
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_96/BiasAdd/ReadVariableOp�
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_96/BiasAdds
dense_96/SeluSeludense_96/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_96/Selu�
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_97/MatMul/ReadVariableOp�
dense_97/MatMulMatMuldense_96/Selu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_97/MatMul�
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_97/BiasAdd/ReadVariableOp�
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_97/BiasAdds
dense_97/SeluSeludense_97/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_97/Selu�
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_98/MatMul/ReadVariableOp�
dense_98/MatMulMatMuldense_97/Selu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_98/MatMul�
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_98/BiasAdd/ReadVariableOp�
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_98/BiasAdds
dense_98/SeluSeludense_98/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_98/Selu�
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_99/MatMul/ReadVariableOp�
dense_99/MatMulMatMuldense_98/Selu:activations:0&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_99/MatMul�
dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_99/BiasAdd/ReadVariableOp�
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_99/BiasAddm
IdentityIdentitydense_99/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������:::::::::::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_96_layer_call_and_return_conditional_losses_6123911

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Seluf
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124141

inputs4
0normalization_24_reshape_readvariableop_resource6
2normalization_24_reshape_1_readvariableop_resource
dense_96_6124120
dense_96_6124122
dense_97_6124125
dense_97_6124127
dense_98_6124130
dense_98_6124132
dense_99_6124135
dense_99_6124137
identity�� dense_96/StatefulPartitionedCall� dense_97/StatefulPartitionedCall� dense_98/StatefulPartitionedCall� dense_99/StatefulPartitionedCall�
'normalization_24/Reshape/ReadVariableOpReadVariableOp0normalization_24_reshape_readvariableop_resource*
_output_shapes
:*
dtype02)
'normalization_24/Reshape/ReadVariableOp�
normalization_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_24/Reshape/shape�
normalization_24/ReshapeReshape/normalization_24/Reshape/ReadVariableOp:value:0'normalization_24/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape�
)normalization_24/Reshape_1/ReadVariableOpReadVariableOp2normalization_24_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)normalization_24/Reshape_1/ReadVariableOp�
 normalization_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_24/Reshape_1/shape�
normalization_24/Reshape_1Reshape1normalization_24/Reshape_1/ReadVariableOp:value:0)normalization_24/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape_1�
normalization_24/subSubinputs!normalization_24/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_24/sub�
normalization_24/SqrtSqrt#normalization_24/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_24/Sqrt�
normalization_24/truedivRealDivnormalization_24/sub:z:0normalization_24/Sqrt:y:0*
T0*'
_output_shapes
:���������2
normalization_24/truediv�
 dense_96/StatefulPartitionedCallStatefulPartitionedCallnormalization_24/truediv:z:0dense_96_6124120dense_96_6124122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_96_layer_call_and_return_conditional_losses_61239112"
 dense_96/StatefulPartitionedCall�
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_6124125dense_97_6124127*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_97_layer_call_and_return_conditional_losses_61239382"
 dense_97/StatefulPartitionedCall�
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_6124130dense_98_6124132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_98_layer_call_and_return_conditional_losses_61239652"
 dense_98/StatefulPartitionedCall�
 dense_99/StatefulPartitionedCallStatefulPartitionedCall)dense_98/StatefulPartitionedCall:output:0dense_99_6124135dense_99_6124137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_99_layer_call_and_return_conditional_losses_61239912"
 dense_99/StatefulPartitionedCall�
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_97_layer_call_and_return_conditional_losses_6124364

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Seluf
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
/__inference_sequential_24_layer_call_fn_6124164
normalization_24_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_61241412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:������������������
0
_user_specified_namenormalization_24_input
�*
�
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124241

inputs4
0normalization_24_reshape_readvariableop_resource6
2normalization_24_reshape_1_readvariableop_resource+
'dense_96_matmul_readvariableop_resource,
(dense_96_biasadd_readvariableop_resource+
'dense_97_matmul_readvariableop_resource,
(dense_97_biasadd_readvariableop_resource+
'dense_98_matmul_readvariableop_resource,
(dense_98_biasadd_readvariableop_resource+
'dense_99_matmul_readvariableop_resource,
(dense_99_biasadd_readvariableop_resource
identity��
'normalization_24/Reshape/ReadVariableOpReadVariableOp0normalization_24_reshape_readvariableop_resource*
_output_shapes
:*
dtype02)
'normalization_24/Reshape/ReadVariableOp�
normalization_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_24/Reshape/shape�
normalization_24/ReshapeReshape/normalization_24/Reshape/ReadVariableOp:value:0'normalization_24/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape�
)normalization_24/Reshape_1/ReadVariableOpReadVariableOp2normalization_24_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)normalization_24/Reshape_1/ReadVariableOp�
 normalization_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_24/Reshape_1/shape�
normalization_24/Reshape_1Reshape1normalization_24/Reshape_1/ReadVariableOp:value:0)normalization_24/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape_1�
normalization_24/subSubinputs!normalization_24/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_24/sub�
normalization_24/SqrtSqrt#normalization_24/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_24/Sqrt�
normalization_24/truedivRealDivnormalization_24/sub:z:0normalization_24/Sqrt:y:0*
T0*'
_output_shapes
:���������2
normalization_24/truediv�
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_96/MatMul/ReadVariableOp�
dense_96/MatMulMatMulnormalization_24/truediv:z:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_96/MatMul�
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_96/BiasAdd/ReadVariableOp�
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_96/BiasAdds
dense_96/SeluSeludense_96/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_96/Selu�
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_97/MatMul/ReadVariableOp�
dense_97/MatMulMatMuldense_96/Selu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_97/MatMul�
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_97/BiasAdd/ReadVariableOp�
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_97/BiasAdds
dense_97/SeluSeludense_97/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_97/Selu�
dense_98/MatMul/ReadVariableOpReadVariableOp'dense_98_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02 
dense_98/MatMul/ReadVariableOp�
dense_98/MatMulMatMuldense_97/Selu:activations:0&dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_98/MatMul�
dense_98/BiasAdd/ReadVariableOpReadVariableOp(dense_98_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_98/BiasAdd/ReadVariableOp�
dense_98/BiasAddBiasAdddense_98/MatMul:product:0'dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_98/BiasAdds
dense_98/SeluSeludense_98/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_98/Selu�
dense_99/MatMul/ReadVariableOpReadVariableOp'dense_99_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_99/MatMul/ReadVariableOp�
dense_99/MatMulMatMuldense_98/Selu:activations:0&dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_99/MatMul�
dense_99/BiasAdd/ReadVariableOpReadVariableOp(dense_99_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_99/BiasAdd/ReadVariableOp�
dense_99/BiasAddBiasAdddense_99/MatMul:product:0'dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_99/BiasAddm
IdentityIdentitydense_99/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������:::::::::::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_98_layer_call_and_return_conditional_losses_6123965

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Seluf
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�$
�
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124043
normalization_24_input4
0normalization_24_reshape_readvariableop_resource6
2normalization_24_reshape_1_readvariableop_resource
dense_96_6124022
dense_96_6124024
dense_97_6124027
dense_97_6124029
dense_98_6124032
dense_98_6124034
dense_99_6124037
dense_99_6124039
identity�� dense_96/StatefulPartitionedCall� dense_97/StatefulPartitionedCall� dense_98/StatefulPartitionedCall� dense_99/StatefulPartitionedCall�
'normalization_24/Reshape/ReadVariableOpReadVariableOp0normalization_24_reshape_readvariableop_resource*
_output_shapes
:*
dtype02)
'normalization_24/Reshape/ReadVariableOp�
normalization_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_24/Reshape/shape�
normalization_24/ReshapeReshape/normalization_24/Reshape/ReadVariableOp:value:0'normalization_24/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape�
)normalization_24/Reshape_1/ReadVariableOpReadVariableOp2normalization_24_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)normalization_24/Reshape_1/ReadVariableOp�
 normalization_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_24/Reshape_1/shape�
normalization_24/Reshape_1Reshape1normalization_24/Reshape_1/ReadVariableOp:value:0)normalization_24/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape_1�
normalization_24/subSubnormalization_24_input!normalization_24/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_24/sub�
normalization_24/SqrtSqrt#normalization_24/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_24/Sqrt�
normalization_24/truedivRealDivnormalization_24/sub:z:0normalization_24/Sqrt:y:0*
T0*'
_output_shapes
:���������2
normalization_24/truediv�
 dense_96/StatefulPartitionedCallStatefulPartitionedCallnormalization_24/truediv:z:0dense_96_6124022dense_96_6124024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_96_layer_call_and_return_conditional_losses_61239112"
 dense_96/StatefulPartitionedCall�
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_6124027dense_97_6124029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_97_layer_call_and_return_conditional_losses_61239382"
 dense_97/StatefulPartitionedCall�
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_6124032dense_98_6124034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_98_layer_call_and_return_conditional_losses_61239652"
 dense_98/StatefulPartitionedCall�
 dense_99/StatefulPartitionedCallStatefulPartitionedCall)dense_98/StatefulPartitionedCall:output:0dense_99_6124037dense_99_6124039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_99_layer_call_and_return_conditional_losses_61239912"
 dense_99/StatefulPartitionedCall�
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:h d
0
_output_shapes
:������������������
0
_user_specified_namenormalization_24_input
�

*__inference_dense_97_layer_call_fn_6124373

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_97_layer_call_and_return_conditional_losses_61239382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

*__inference_dense_98_layer_call_fn_6124393

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_98_layer_call_and_return_conditional_losses_61239652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_dense_98_layer_call_and_return_conditional_losses_6124384

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Seluf
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�6
�
"__inference__wrapped_model_6123885
normalization_24_inputB
>sequential_24_normalization_24_reshape_readvariableop_resourceD
@sequential_24_normalization_24_reshape_1_readvariableop_resource9
5sequential_24_dense_96_matmul_readvariableop_resource:
6sequential_24_dense_96_biasadd_readvariableop_resource9
5sequential_24_dense_97_matmul_readvariableop_resource:
6sequential_24_dense_97_biasadd_readvariableop_resource9
5sequential_24_dense_98_matmul_readvariableop_resource:
6sequential_24_dense_98_biasadd_readvariableop_resource9
5sequential_24_dense_99_matmul_readvariableop_resource:
6sequential_24_dense_99_biasadd_readvariableop_resource
identity��
5sequential_24/normalization_24/Reshape/ReadVariableOpReadVariableOp>sequential_24_normalization_24_reshape_readvariableop_resource*
_output_shapes
:*
dtype027
5sequential_24/normalization_24/Reshape/ReadVariableOp�
,sequential_24/normalization_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2.
,sequential_24/normalization_24/Reshape/shape�
&sequential_24/normalization_24/ReshapeReshape=sequential_24/normalization_24/Reshape/ReadVariableOp:value:05sequential_24/normalization_24/Reshape/shape:output:0*
T0*
_output_shapes

:2(
&sequential_24/normalization_24/Reshape�
7sequential_24/normalization_24/Reshape_1/ReadVariableOpReadVariableOp@sequential_24_normalization_24_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype029
7sequential_24/normalization_24/Reshape_1/ReadVariableOp�
.sequential_24/normalization_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      20
.sequential_24/normalization_24/Reshape_1/shape�
(sequential_24/normalization_24/Reshape_1Reshape?sequential_24/normalization_24/Reshape_1/ReadVariableOp:value:07sequential_24/normalization_24/Reshape_1/shape:output:0*
T0*
_output_shapes

:2*
(sequential_24/normalization_24/Reshape_1�
"sequential_24/normalization_24/subSubnormalization_24_input/sequential_24/normalization_24/Reshape:output:0*
T0*'
_output_shapes
:���������2$
"sequential_24/normalization_24/sub�
#sequential_24/normalization_24/SqrtSqrt1sequential_24/normalization_24/Reshape_1:output:0*
T0*
_output_shapes

:2%
#sequential_24/normalization_24/Sqrt�
&sequential_24/normalization_24/truedivRealDiv&sequential_24/normalization_24/sub:z:0'sequential_24/normalization_24/Sqrt:y:0*
T0*'
_output_shapes
:���������2(
&sequential_24/normalization_24/truediv�
,sequential_24/dense_96/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_96_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,sequential_24/dense_96/MatMul/ReadVariableOp�
sequential_24/dense_96/MatMulMatMul*sequential_24/normalization_24/truediv:z:04sequential_24/dense_96/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_24/dense_96/MatMul�
-sequential_24/dense_96/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_96_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_24/dense_96/BiasAdd/ReadVariableOp�
sequential_24/dense_96/BiasAddBiasAdd'sequential_24/dense_96/MatMul:product:05sequential_24/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
sequential_24/dense_96/BiasAdd�
sequential_24/dense_96/SeluSelu'sequential_24/dense_96/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
sequential_24/dense_96/Selu�
,sequential_24/dense_97/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_97_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02.
,sequential_24/dense_97/MatMul/ReadVariableOp�
sequential_24/dense_97/MatMulMatMul)sequential_24/dense_96/Selu:activations:04sequential_24/dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_24/dense_97/MatMul�
-sequential_24/dense_97/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_24/dense_97/BiasAdd/ReadVariableOp�
sequential_24/dense_97/BiasAddBiasAdd'sequential_24/dense_97/MatMul:product:05sequential_24/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
sequential_24/dense_97/BiasAdd�
sequential_24/dense_97/SeluSelu'sequential_24/dense_97/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
sequential_24/dense_97/Selu�
,sequential_24/dense_98/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_98_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02.
,sequential_24/dense_98/MatMul/ReadVariableOp�
sequential_24/dense_98/MatMulMatMul)sequential_24/dense_97/Selu:activations:04sequential_24/dense_98/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_24/dense_98/MatMul�
-sequential_24/dense_98/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_98_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_24/dense_98/BiasAdd/ReadVariableOp�
sequential_24/dense_98/BiasAddBiasAdd'sequential_24/dense_98/MatMul:product:05sequential_24/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
sequential_24/dense_98/BiasAdd�
sequential_24/dense_98/SeluSelu'sequential_24/dense_98/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
sequential_24/dense_98/Selu�
,sequential_24/dense_99/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_99_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,sequential_24/dense_99/MatMul/ReadVariableOp�
sequential_24/dense_99/MatMulMatMul)sequential_24/dense_98/Selu:activations:04sequential_24/dense_99/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_24/dense_99/MatMul�
-sequential_24/dense_99/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_99_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_24/dense_99/BiasAdd/ReadVariableOp�
sequential_24/dense_99/BiasAddBiasAdd'sequential_24/dense_99/MatMul:product:05sequential_24/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_24/dense_99/BiasAdd{
IdentityIdentity'sequential_24/dense_99/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������:::::::::::h d
0
_output_shapes
:������������������
0
_user_specified_namenormalization_24_input
�

*__inference_dense_96_layer_call_fn_6124353

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_96_layer_call_and_return_conditional_losses_61239112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_24_layer_call_fn_6124308

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_61240812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
/__inference_sequential_24_layer_call_fn_6124333

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_61241412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
#__inference__traced_restore_6124655
file_prefix
assignvariableop_mean
assignvariableop_1_variance
assignvariableop_2_count&
"assignvariableop_3_dense_96_kernel$
 assignvariableop_4_dense_96_bias&
"assignvariableop_5_dense_97_kernel$
 assignvariableop_6_dense_97_bias&
"assignvariableop_7_dense_98_kernel$
 assignvariableop_8_dense_98_bias&
"assignvariableop_9_dense_99_kernel%
!assignvariableop_10_dense_99_bias"
assignvariableop_11_nadam_iter$
 assignvariableop_12_nadam_beta_1$
 assignvariableop_13_nadam_beta_2#
assignvariableop_14_nadam_decay+
'assignvariableop_15_nadam_learning_rate,
(assignvariableop_16_nadam_momentum_cache
assignvariableop_17_total
assignvariableop_18_count_1/
+assignvariableop_19_nadam_dense_96_kernel_m-
)assignvariableop_20_nadam_dense_96_bias_m/
+assignvariableop_21_nadam_dense_97_kernel_m-
)assignvariableop_22_nadam_dense_97_bias_m/
+assignvariableop_23_nadam_dense_98_kernel_m-
)assignvariableop_24_nadam_dense_98_bias_m/
+assignvariableop_25_nadam_dense_99_kernel_m-
)assignvariableop_26_nadam_dense_99_bias_m/
+assignvariableop_27_nadam_dense_96_kernel_v-
)assignvariableop_28_nadam_dense_96_bias_v/
+assignvariableop_29_nadam_dense_97_kernel_v-
)assignvariableop_30_nadam_dense_97_bias_v/
+assignvariableop_31_nadam_dense_98_kernel_v-
)assignvariableop_32_nadam_dense_98_bias_v/
+assignvariableop_33_nadam_dense_99_kernel_v-
)assignvariableop_34_nadam_dense_99_bias_v
identity_36��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_96_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_96_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_97_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_97_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_98_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_98_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_99_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_99_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_nadam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp assignvariableop_12_nadam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp assignvariableop_13_nadam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_nadam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_nadam_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_nadam_momentum_cacheIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_nadam_dense_96_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_nadam_dense_96_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp+assignvariableop_21_nadam_dense_97_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_nadam_dense_97_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_nadam_dense_98_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_nadam_dense_98_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_nadam_dense_99_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_nadam_dense_99_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_nadam_dense_96_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_nadam_dense_96_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_nadam_dense_97_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_nadam_dense_97_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_nadam_dense_98_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_nadam_dense_98_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_nadam_dense_99_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_nadam_dense_99_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35�
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
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
�#
�
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124081

inputs4
0normalization_24_reshape_readvariableop_resource6
2normalization_24_reshape_1_readvariableop_resource
dense_96_6124060
dense_96_6124062
dense_97_6124065
dense_97_6124067
dense_98_6124070
dense_98_6124072
dense_99_6124075
dense_99_6124077
identity�� dense_96/StatefulPartitionedCall� dense_97/StatefulPartitionedCall� dense_98/StatefulPartitionedCall� dense_99/StatefulPartitionedCall�
'normalization_24/Reshape/ReadVariableOpReadVariableOp0normalization_24_reshape_readvariableop_resource*
_output_shapes
:*
dtype02)
'normalization_24/Reshape/ReadVariableOp�
normalization_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_24/Reshape/shape�
normalization_24/ReshapeReshape/normalization_24/Reshape/ReadVariableOp:value:0'normalization_24/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape�
)normalization_24/Reshape_1/ReadVariableOpReadVariableOp2normalization_24_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)normalization_24/Reshape_1/ReadVariableOp�
 normalization_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_24/Reshape_1/shape�
normalization_24/Reshape_1Reshape1normalization_24/Reshape_1/ReadVariableOp:value:0)normalization_24/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape_1�
normalization_24/subSubinputs!normalization_24/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_24/sub�
normalization_24/SqrtSqrt#normalization_24/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_24/Sqrt�
normalization_24/truedivRealDivnormalization_24/sub:z:0normalization_24/Sqrt:y:0*
T0*'
_output_shapes
:���������2
normalization_24/truediv�
 dense_96/StatefulPartitionedCallStatefulPartitionedCallnormalization_24/truediv:z:0dense_96_6124060dense_96_6124062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_96_layer_call_and_return_conditional_losses_61239112"
 dense_96/StatefulPartitionedCall�
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_6124065dense_97_6124067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_97_layer_call_and_return_conditional_losses_61239382"
 dense_97/StatefulPartitionedCall�
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_6124070dense_98_6124072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_98_layer_call_and_return_conditional_losses_61239652"
 dense_98/StatefulPartitionedCall�
 dense_99/StatefulPartitionedCallStatefulPartitionedCall)dense_98/StatefulPartitionedCall:output:0dense_99_6124075dense_99_6124077*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_99_layer_call_and_return_conditional_losses_61239912"
 dense_99/StatefulPartitionedCall�
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_dense_99_layer_call_and_return_conditional_losses_6123991

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_dense_96_layer_call_and_return_conditional_losses_6124344

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Seluf
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

*__inference_dense_99_layer_call_fn_6124412

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_99_layer_call_and_return_conditional_losses_61239912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_dense_99_layer_call_and_return_conditional_losses_6124403

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�$
�
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124008
normalization_24_input4
0normalization_24_reshape_readvariableop_resource6
2normalization_24_reshape_1_readvariableop_resource
dense_96_6123922
dense_96_6123924
dense_97_6123949
dense_97_6123951
dense_98_6123976
dense_98_6123978
dense_99_6124002
dense_99_6124004
identity�� dense_96/StatefulPartitionedCall� dense_97/StatefulPartitionedCall� dense_98/StatefulPartitionedCall� dense_99/StatefulPartitionedCall�
'normalization_24/Reshape/ReadVariableOpReadVariableOp0normalization_24_reshape_readvariableop_resource*
_output_shapes
:*
dtype02)
'normalization_24/Reshape/ReadVariableOp�
normalization_24/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2 
normalization_24/Reshape/shape�
normalization_24/ReshapeReshape/normalization_24/Reshape/ReadVariableOp:value:0'normalization_24/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape�
)normalization_24/Reshape_1/ReadVariableOpReadVariableOp2normalization_24_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)normalization_24/Reshape_1/ReadVariableOp�
 normalization_24/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2"
 normalization_24/Reshape_1/shape�
normalization_24/Reshape_1Reshape1normalization_24/Reshape_1/ReadVariableOp:value:0)normalization_24/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization_24/Reshape_1�
normalization_24/subSubnormalization_24_input!normalization_24/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization_24/sub�
normalization_24/SqrtSqrt#normalization_24/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization_24/Sqrt�
normalization_24/truedivRealDivnormalization_24/sub:z:0normalization_24/Sqrt:y:0*
T0*'
_output_shapes
:���������2
normalization_24/truediv�
 dense_96/StatefulPartitionedCallStatefulPartitionedCallnormalization_24/truediv:z:0dense_96_6123922dense_96_6123924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_96_layer_call_and_return_conditional_losses_61239112"
 dense_96/StatefulPartitionedCall�
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_6123949dense_97_6123951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_97_layer_call_and_return_conditional_losses_61239382"
 dense_97/StatefulPartitionedCall�
 dense_98/StatefulPartitionedCallStatefulPartitionedCall)dense_97/StatefulPartitionedCall:output:0dense_98_6123976dense_98_6123978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_98_layer_call_and_return_conditional_losses_61239652"
 dense_98/StatefulPartitionedCall�
 dense_99/StatefulPartitionedCallStatefulPartitionedCall)dense_98/StatefulPartitionedCall:output:0dense_99_6124002dense_99_6124004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_99_layer_call_and_return_conditional_losses_61239912"
 dense_99/StatefulPartitionedCall�
IdentityIdentity)dense_99/StatefulPartitionedCall:output:0!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall!^dense_98/StatefulPartitionedCall!^dense_99/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:������������������::::::::::2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2D
 dense_98/StatefulPartitionedCall dense_98/StatefulPartitionedCall2D
 dense_99/StatefulPartitionedCall dense_99/StatefulPartitionedCall:h d
0
_output_shapes
:������������������
0
_user_specified_namenormalization_24_input
�
�
E__inference_dense_97_layer_call_and_return_conditional_losses_6123938

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Seluf
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:::O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�K
�
 __inference__traced_save_6124540
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	.
*savev2_dense_96_kernel_read_readvariableop,
(savev2_dense_96_bias_read_readvariableop.
*savev2_dense_97_kernel_read_readvariableop,
(savev2_dense_97_bias_read_readvariableop.
*savev2_dense_98_kernel_read_readvariableop,
(savev2_dense_98_bias_read_readvariableop.
*savev2_dense_99_kernel_read_readvariableop,
(savev2_dense_99_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_nadam_dense_96_kernel_m_read_readvariableop4
0savev2_nadam_dense_96_bias_m_read_readvariableop6
2savev2_nadam_dense_97_kernel_m_read_readvariableop4
0savev2_nadam_dense_97_bias_m_read_readvariableop6
2savev2_nadam_dense_98_kernel_m_read_readvariableop4
0savev2_nadam_dense_98_bias_m_read_readvariableop6
2savev2_nadam_dense_99_kernel_m_read_readvariableop4
0savev2_nadam_dense_99_bias_m_read_readvariableop6
2savev2_nadam_dense_96_kernel_v_read_readvariableop4
0savev2_nadam_dense_96_bias_v_read_readvariableop6
2savev2_nadam_dense_97_kernel_v_read_readvariableop4
0savev2_nadam_dense_97_bias_v_read_readvariableop6
2savev2_nadam_dense_98_kernel_v_read_readvariableop4
0savev2_nadam_dense_98_bias_v_read_readvariableop6
2savev2_nadam_dense_99_kernel_v_read_readvariableop4
0savev2_nadam_dense_99_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_44e43664a4264d63849a1fccfcfe9ea8/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*�
value�B�$B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop*savev2_dense_96_kernel_read_readvariableop(savev2_dense_96_bias_read_readvariableop*savev2_dense_97_kernel_read_readvariableop(savev2_dense_97_bias_read_readvariableop*savev2_dense_98_kernel_read_readvariableop(savev2_dense_98_bias_read_readvariableop*savev2_dense_99_kernel_read_readvariableop(savev2_dense_99_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop2savev2_nadam_dense_96_kernel_m_read_readvariableop0savev2_nadam_dense_96_bias_m_read_readvariableop2savev2_nadam_dense_97_kernel_m_read_readvariableop0savev2_nadam_dense_97_bias_m_read_readvariableop2savev2_nadam_dense_98_kernel_m_read_readvariableop0savev2_nadam_dense_98_bias_m_read_readvariableop2savev2_nadam_dense_99_kernel_m_read_readvariableop0savev2_nadam_dense_99_bias_m_read_readvariableop2savev2_nadam_dense_96_kernel_v_read_readvariableop0savev2_nadam_dense_96_bias_v_read_readvariableop2savev2_nadam_dense_97_kernel_v_read_readvariableop0savev2_nadam_dense_97_bias_v_read_readvariableop2savev2_nadam_dense_98_kernel_v_read_readvariableop0savev2_nadam_dense_98_bias_v_read_readvariableop2savev2_nadam_dense_99_kernel_v_read_readvariableop0savev2_nadam_dense_99_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$		2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::: :
:
:

:
:

:
:
:: : : : : : : : :
:
:

:
:

:
:
::
:
:

:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 	

_output_shapes
:
:$
 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$  

_output_shapes

:

: !

_output_shapes
:
:$" 

_output_shapes

:
: #

_output_shapes
::$

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
b
normalization_24_inputH
(serving_default_normalization_24_input:0������������������<
dense_990
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�+
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
^_default_save_signature
*_&call_and_return_all_conditional_losses
`__call__"�(
_tf_keras_sequential�'{"class_name": "Sequential", "name": "sequential_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "normalization_24_input"}}, {"class_name": "Normalization", "config": {"name": "normalization_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}}, {"class_name": "Dense", "config": {"name": "dense_96", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_97", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_98", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_99", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "normalization_24_input"}}, {"class_name": "Normalization", "config": {"name": "normalization_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}}, {"class_name": "Dense", "config": {"name": "dense_96", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_97", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_98", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_99", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "MeanSquaredError", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
�
state_variables
_broadcast_shape
mean
variance
	count
	keras_api"�
_tf_keras_layer�{"class_name": "Normalization", "name": "normalization_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": [512, 8]}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_96", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_96", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_97", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_97", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*e&call_and_return_all_conditional_losses
f__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_98", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_98", "trainable": true, "dtype": "float32", "units": 10, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*g&call_and_return_all_conditional_losses
h__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_99", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_99", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�
*iter

+beta_1

,beta_2
	-decay
.learning_rate
/momentum_cachemNmOmPmQmRmS$mT%mUvVvWvXvYvZv[$v\%v]"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
$6
%7"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
8
$9
%10"
trackable_list_wrapper
�
0non_trainable_variables
1layer_metrics
regularization_losses
2layer_regularization_losses
3metrics

4layers
trainable_variables
		variables
`__call__
^_default_save_signature
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
,
iserving_default"
signature_map
C
mean
variance
	count"
trackable_dict_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
!:
2dense_96/kernel
:
2dense_96/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
5non_trainable_variables
6layer_metrics
regularization_losses
7metrics
8layer_regularization_losses

9layers
	variables
trainable_variables
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
!:

2dense_97/kernel
:
2dense_97/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
:non_trainable_variables
;layer_metrics
regularization_losses
<metrics
=layer_regularization_losses

>layers
	variables
trainable_variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
!:

2dense_98/kernel
:
2dense_98/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
?non_trainable_variables
@layer_metrics
 regularization_losses
Ametrics
Blayer_regularization_losses

Clayers
!	variables
"trainable_variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_99/kernel
:2dense_99/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
Dnon_trainable_variables
Elayer_metrics
&regularization_losses
Fmetrics
Glayer_regularization_losses

Hlayers
'	variables
(trainable_variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
C
0
1
2
3
4"
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
�
	Jtotal
	Kcount
L	variables
M	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
J0
K1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
':%
2Nadam/dense_96/kernel/m
!:
2Nadam/dense_96/bias/m
':%

2Nadam/dense_97/kernel/m
!:
2Nadam/dense_97/bias/m
':%

2Nadam/dense_98/kernel/m
!:
2Nadam/dense_98/bias/m
':%
2Nadam/dense_99/kernel/m
!:2Nadam/dense_99/bias/m
':%
2Nadam/dense_96/kernel/v
!:
2Nadam/dense_96/bias/v
':%

2Nadam/dense_97/kernel/v
!:
2Nadam/dense_97/bias/v
':%

2Nadam/dense_98/kernel/v
!:
2Nadam/dense_98/bias/v
':%
2Nadam/dense_99/kernel/v
!:2Nadam/dense_99/bias/v
�2�
"__inference__wrapped_model_6123885�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *>�;
9�6
normalization_24_input������������������
�2�
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124241
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124283
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124008
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124043�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_sequential_24_layer_call_fn_6124104
/__inference_sequential_24_layer_call_fn_6124164
/__inference_sequential_24_layer_call_fn_6124308
/__inference_sequential_24_layer_call_fn_6124333�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dense_96_layer_call_and_return_conditional_losses_6124344�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_96_layer_call_fn_6124353�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_97_layer_call_and_return_conditional_losses_6124364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_97_layer_call_fn_6124373�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_98_layer_call_and_return_conditional_losses_6124384�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_98_layer_call_fn_6124393�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_99_layer_call_and_return_conditional_losses_6124403�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_99_layer_call_fn_6124412�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
CBA
%__inference_signature_wrapper_6124199normalization_24_input�
"__inference__wrapped_model_6123885�
$%H�E
>�;
9�6
normalization_24_input������������������
� "3�0
.
dense_99"�
dense_99����������
E__inference_dense_96_layer_call_and_return_conditional_losses_6124344\/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� }
*__inference_dense_96_layer_call_fn_6124353O/�,
%�"
 �
inputs���������
� "����������
�
E__inference_dense_97_layer_call_and_return_conditional_losses_6124364\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� }
*__inference_dense_97_layer_call_fn_6124373O/�,
%�"
 �
inputs���������

� "����������
�
E__inference_dense_98_layer_call_and_return_conditional_losses_6124384\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� }
*__inference_dense_98_layer_call_fn_6124393O/�,
%�"
 �
inputs���������

� "����������
�
E__inference_dense_99_layer_call_and_return_conditional_losses_6124403\$%/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� }
*__inference_dense_99_layer_call_fn_6124412O$%/�,
%�"
 �
inputs���������

� "�����������
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124008�
$%P�M
F�C
9�6
normalization_24_input������������������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124043�
$%P�M
F�C
9�6
normalization_24_input������������������
p 

 
� "%�"
�
0���������
� �
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124241u
$%@�=
6�3
)�&
inputs������������������
p

 
� "%�"
�
0���������
� �
J__inference_sequential_24_layer_call_and_return_conditional_losses_6124283u
$%@�=
6�3
)�&
inputs������������������
p 

 
� "%�"
�
0���������
� �
/__inference_sequential_24_layer_call_fn_6124104x
$%P�M
F�C
9�6
normalization_24_input������������������
p

 
� "�����������
/__inference_sequential_24_layer_call_fn_6124164x
$%P�M
F�C
9�6
normalization_24_input������������������
p 

 
� "�����������
/__inference_sequential_24_layer_call_fn_6124308h
$%@�=
6�3
)�&
inputs������������������
p

 
� "�����������
/__inference_sequential_24_layer_call_fn_6124333h
$%@�=
6�3
)�&
inputs������������������
p 

 
� "�����������
%__inference_signature_wrapper_6124199�
$%b�_
� 
X�U
S
normalization_24_input9�6
normalization_24_input������������������"3�0
.
dense_99"�
dense_99���������