MOV r8 r0 3
CMPE r0 r8 1
CJMP r0 r0 12
ADD r9 r9 1
AND r7 r8 1
CMPE r0 r7 0
CJMP r0 r0 10
SHL r7 r8 1
ADD r7 r7 r8
ADD r8 r7 1
SHR r8 r8 1
JMP r0 r0 1
ANSWER r0 r0 r9
