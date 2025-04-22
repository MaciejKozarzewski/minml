/*
 * assembly_macros.hpp
 *
 *  Created on: Jun 20, 2023
 *      Author: Maciej Kozarzewski
 *      Greatly inspired by BLIS project https://github.com/flame/blis
 */

#ifndef BACKEND_CPU_ASSEMBLY_MACROS_HPP_
#define BACKEND_CPU_ASSEMBLY_MACROS_HPP_

#define STRINGIFY_(...) #__VA_ARGS__
#define GET_MACRO_(_1_,_2_,_3_,_4_,NAME,...) NAME
#define LABEL_(x) ".L" STRINGIFY_(x) "%="

#define JMP_(insn, target) STRINGIFY_(insn) " " LABEL_(target) "\n\t"
#define INSTR_4_(name,_0,_1,_2,_3) STRINGIFY_(name) " " STRINGIFY_(_0,_1,_2,_3) "\n\t"
#define INSTR_3_(name,_0,_1,_2) STRINGIFY_(name) " " STRINGIFY_(_0,_1,_2) "\n\t"
#define INSTR_2_(name,_0,_1) STRINGIFY_(name) " " STRINGIFY_(_0,_1) "\n\t"
#define INSTR_1_(name,_0) STRINGIFY_(name) " " STRINGIFY_(_0) "\n\t"
#define INSTR_0_(name) STRINGIFY_(name) "\n\t"
#define INSTR_(name,...) GET_MACRO_(__VA_ARGS__,INSTR_4_,INSTR_3_,INSTR_2_,INSTR_1_,INSTR_0_)(name,__VA_ARGS__)

// MEM(rax)            ->     (%rax)        or [rax]
// MEM(rax,0x80)       -> 0x80(%rax)        or [rax + 0x80]
// MEM(rax,rsi,4)      ->     (%rax,%rsi,4) or [rax + rsi*4]
// MEM(rax,rsi,4,0x80) -> 0x80(%rax,%rsi,4) or [rax + rsi*4 + 0x80]
#define MEM_4_(reg,off,scale,disp) (disp)(reg,off,scale)
#define MEM_3_(reg,off,scale) (reg,off,scale)
#define MEM_2_(reg,disp) (disp)(reg)
#define MEM_1_(reg) (reg)

#define REGISTER_(r) %%r

/*
 *
 */

#define begin_asm() __asm__ volatile (
#define end_asm(...) __VA_ARGS__ );

#define label(x) LABEL_(x) ":\n\t"

#define imm(x) $##x
#define var(x) %[x]
#define mask_(x) %{x%}

#define mem(...) GET_MACRO_(__VA_ARGS__,MEM_4_,MEM_3_,MEM_2_,MEM_1_)(__VA_ARGS__)

/*
 * 8-bit registers
 */
#define al REGISTER_(al)
#define ah REGISTER_(ah)
#define bl REGISTER_(bl)
#define bh REGISTER_(bh)
#define cl REGISTER_(cl)
#define ch REGISTER_(ch)
#define dl REGISTER_(dl)
#define dh REGISTER_(dh)
#define r8b REGISTER_(r8b)
#define r9b REGISTER_(r9b)
#define r10b REGISTER_(r10b)
#define r11b REGISTER_(r11b)
#define r12b REGISTER_(r12b)
#define r13b REGISTER_(r13b)
#define r14b REGISTER_(r14b)
#define r15b REGISTER_(r15b)

/*
 * 16-bit registers
 */
#define ax REGISTER_(ax)
#define bx REGISTER_(bx)
#define cx REGISTER_(cx)
#define dx REGISTER_(dx)
#define si REGISTER_(si)
#define di REGISTER_(di)
#define bp REGISTER_(bp)
#define sp REGISTER_(sp)
#define r8w REGISTER_(r8w)
#define r9w REGISTER_(r9w)
#define r10w REGISTER_(r10w)
#define r11w REGISTER_(r11w)
#define r12w REGISTER_(r12w)
#define r13w REGISTER_(r13w)
#define r14w REGISTER_(r14w)
#define r15w REGISTER_(r15w)

/*
 * 32-bit registers
 */
#define eax REGISTER_(eax)
#define ebx REGISTER_(ebx)
#define ecx REGISTER_(ecx)
#define edx REGISTER_(edx)
#define esp REGISTER_(esp)
#define ebp REGISTER_(ebp)
#define edi REGISTER_(edi)
#define esi REGISTER_(esi)
#define r8d REGISTER_(r8d)
#define r9d REGISTER_(r9d)
#define r10d REGISTER_(r10d)
#define r11d REGISTER_(r11d)
#define r12d REGISTER_(r12d)
#define r13d REGISTER_(r13d)
#define r14d REGISTER_(r14d)
#define r15d REGISTER_(r15d)

/*
 * 64-bit registers
 */
#define rax REGISTER_(rax)
#define rbx REGISTER_(rbx)
#define rcx REGISTER_(rcx)
#define rdx REGISTER_(rdx)
#define rsp REGISTER_(rsp)
#define rbp REGISTER_(rbp)
#define rdi REGISTER_(rdi)
#define rsi REGISTER_(rsi)
#define r8 REGISTER_(r8)
#define r9 REGISTER_(r9)
#define r10 REGISTER_(r10)
#define r11 REGISTER_(r11)
#define r12 REGISTER_(r12)
#define r13 REGISTER_(r13)
#define r14 REGISTER_(r14)
#define r15 REGISTER_(r15)

/*
 * Vector registers
 */
#define xmm(x) REGISTER_(Xmm##x)
#define ymm(x) REGISTER_(Ymm##x)
#define zmm(x) REGISTER_(Zmm##x)
#define k(x) REGISTER_(k##x)
#define mask_k(x) mask_(k(x))
#define mask_kz(x) mask_(k(x))mask_(z)

/*
 * 128-bit SSE registers
 */
#define xmm0 xmm(0)
#define xmm1 xmm(1)
#define xmm2 xmm(2)
#define xmm3 xmm(3)
#define xmm4 xmm(4)
#define xmm5 xmm(5)
#define xmm6 xmm(6)
#define xmm7 xmm(7)
#define xmm8 xmm(8)
#define xmm9 xmm(9)
#define xmm10 xmm(10)
#define xmm11 xmm(11)
#define xmm12 xmm(12)
#define xmm13 xmm(13)
#define xmm14 xmm(14)
#define xmm15 xmm(15)
#define xmm16 xmm(16)
#define xmm17 xmm(17)
#define xmm18 xmm(18)
#define xmm19 xmm(19)
#define xmm20 xmm(20)
#define xmm21 xmm(21)
#define xmm22 xmm(22)
#define xmm23 xmm(23)
#define xmm24 xmm(24)
#define xmm25 xmm(25)
#define xmm26 xmm(26)
#define xmm27 xmm(27)
#define xmm28 xmm(28)
#define xmm29 xmm(29)
#define xmm30 xmm(30)
#define xmm31 xmm(31)

/*
 * 256-bit AVX registers
 */
#define ymm0 ymm(0)
#define ymm1 ymm(1)
#define ymm2 ymm(2)
#define ymm3 ymm(3)
#define ymm4 ymm(4)
#define ymm5 ymm(5)
#define ymm6 ymm(6)
#define ymm7 ymm(7)
#define ymm8 ymm(8)
#define ymm9 ymm(9)
#define ymm10 ymm(10)
#define ymm11 ymm(11)
#define ymm12 ymm(12)
#define ymm13 ymm(13)
#define ymm14 ymm(14)
#define ymm15 ymm(15)
#define ymm16 ymm(16)
#define ymm17 ymm(17)
#define ymm18 ymm(18)
#define ymm19 ymm(19)
#define ymm20 ymm(20)
#define ymm21 ymm(21)
#define ymm22 ymm(22)
#define ymm23 ymm(23)
#define ymm24 ymm(24)
#define ymm25 ymm(25)
#define ymm26 ymm(26)
#define ymm27 ymm(27)
#define ymm28 ymm(28)
#define ymm29 ymm(29)
#define ymm30 ymm(30)
#define ymm31 ymm(31)

/*
 * 512-bit AVX512 registers
 */
#define zmm0 zmm(0)
#define zmm1 zmm(1)
#define zmm2 zmm(2)
#define zmm3 zmm(3)
#define zmm4 zmm(4)
#define zmm5 zmm(5)
#define zmm6 zmm(6)
#define zmm7 zmm(7)
#define zmm8 zmm(8)
#define zmm9 zmm(9)
#define zmm10 zmm(10)
#define zmm11 zmm(11)
#define zmm12 zmm(12)
#define zmm13 zmm(13)
#define zmm14 zmm(14)
#define zmm15 zmm(15)
#define zmm16 zmm(16)
#define zmm17 zmm(17)
#define zmm18 zmm(18)
#define zmm19 zmm(19)
#define zmm20 zmm(20)
#define zmm21 zmm(21)
#define zmm22 zmm(22)
#define zmm23 zmm(23)
#define zmm24 zmm(24)
#define zmm25 zmm(25)
#define zmm26 zmm(26)
#define zmm27 zmm(27)
#define zmm28 zmm(28)
#define zmm29 zmm(29)
#define zmm30 zmm(30)
#define zmm31 zmm(31)

/*
 * Jumps
 */
#define jc(_0) JMP_(jc, _0)
#define jb(_0) jc(_0)
#define jnae(_0) jc(_0)
#define jnc(_0) JMP_(jnc, _0)
#define jnb(_0) JNC(_0)
#define jae(_0) JNC(_0)

#define jo(_0) JMP_(jo, _0)
#define jno(_0) JMP_(jno, _0)

#define jp(_0) JMP_(jp, _0)
#define jpe(_0) jp(_0)
#define jnp(_0) JMP_(jnp, _0)
#define jpo(_0) JNP(_0)

#define js(_0) JMP_(js, _0)
#define jns(_0) JMP_(jns, _0)

#define ja(_0) JMP_(ja, _0)
#define jnbe(_0) ja(_0)
#define jna(_0) JMP_(jna, _0)
#define jbe(_0) JNA(_0)

#define jl(_0) JMP_(jl, _0)
#define jnge(_0) jl(_0)
#define jnl(_0) JMP_(jnl, _0)
#define jge(_0) JNL(_0)

#define jg(_0) JMP_(jg, _0)
#define jnle(_0) jg(_0)
#define jng(_0) JMP_(jng, _0)
#define jle(_0) JNG(_0)

#define je(_0) JMP_(je, _0)
#define jz(_0) je(_0)
#define jne(_0) JMP_(jne, _0)
#define jnz(_0) JNE(_0)

#define jmp(_0) JMP_(jmp, _0)

#define sete(_0) INSTR_(sete, _0)
#define setz(_0) sete(_0)

/*
 * Comparisons
 */
#define cmp(_0, _1) INSTR_(cmp, _0, _1)
#define test(_0, _1) INSTR_(test, _0, _1)

/*
 * Integer math
 */
#define and_(_0, _1) INSTR_(and, _0, _1)
#define or_(_0, _1) INSTR_(or, _0, _1)
#define xor_(_0, _1) INSTR_(xor, _0, _1)
#define add(_0, _1) INSTR_(add, _0, _1)
#define sub(_0, _1) INSTR_(sub, _0, _1)
#define imul(_0, _1) INSTR_(imul, _0, _1)
#define sal(...) INSTR_(sal, __VA_ARGS__)
#define sar(...) INSTR_(sar, __VA_ARGS__)
#define shlx(_0, _1, _2) INSTR_(shlx, _0, _1, _2)
#define shrx(_0, _1, _2) INSTR_(shrx, _0, _1, _2)
#define rorx(_0, _1, _2) INSTR_(rorx, _0, _1, _2)
#define dec(_0) INSTR_(dec, _0)
#define inc(_0) INSTR_(inc, _0)

/*
 * Memory access
 */
#define lea(_0, _1) INSTR_(lea, _0, _1)
#define mov(_0, _1) INSTR_(mov, _0, _1)
#define movzw(_0, _1) INSTR_(movzw, _0, _1)
#define movd(_0, _1) INSTR_(movd, _0, _1)
#define movl(_0, _1) INSTR_(movl, _0, _1)
#define movq(_0, _1) INSTR_(movq, _0, _1)
#define cmova(_0, _1) INSTR_(cmova, _0, _1)
#define cmovae(_0, _1) INSTR_(cmovae, _0, _1)
#define cmovb(_0, _1) INSTR_(cmovb, _0, _1)
#define cmovbe(_0, _1) INSTR_(cmovbe, _0, _1)
#define cmovc(_0, _1) INSTR_(cmovc, _0, _1)
#define cmovp(_0, _1) INSTR_(cmovp, _0, _1)
#define cmovo(_0, _1) INSTR_(cmovo, _0, _1)
#define cmovs(_0, _1) INSTR_(cmovs, _0, _1)
#define cmove(_0, _1) INSTR_(cmove, _0, _1)
#define cmovz(_0, _1) INSTR_(cmovz, _0, _1)
#define cmovg(_0, _1) INSTR_(cmovg, _0, _1)
#define cmovge(_0, _1) INSTR_(cmovge, _0, _1)
#define cmovl(_0, _1) INSTR_(cmovl, _0, _1)
#define cmovle(_0, _1) INSTR_(cmovle, _0, _1)
#define cmovna(_0, _1) INSTR_(cmovna, _0, _1)
#define cmovnae(_0, _1) INSTR_(cmovnae, _0, _1)
#define cmovnb(_0, _1) INSTR_(cmovnb, _0, _1)
#define cmovnbe(_0, _1) INSTR_(cmovnbe, _0, _1)
#define cmovnc(_0, _1) INSTR_(cmovnc, _0, _1)
#define cmovnp(_0, _1) INSTR_(cmovnp, _0, _1)
#define cmovno(_0, _1) INSTR_(cmovno, _0, _1)
#define cmovns(_0, _1) INSTR_(cmovns, _0, _1)
#define cmovne(_0, _1) INSTR_(cmovne, _0, _1)
#define cmovnz(_0, _1) INSTR_(cmovnz, _0, _1)
#define cmovng(_0, _1) INSTR_(cmovng, _0, _1)
#define cmovnge(_0, _1) INSTR_(cmovnge, _0, _1)
#define cmovnl(_0, _1) INSTR_(cmovnl, _0, _1)
#define cmovnle(_0, _1) INSTR_(cmovnle, _0, _1)
#define kmovw(_0, _1) INSTR_(kmovw, _0, _1)

/*
 * Vector moves
 */
#define movss(_0, _1) INSTR_(movss, _0, _1)
#define movsd(_0, _1) INSTR_(movsd, _0, _1)
#define movaps(_0, _1) INSTR_(movaps, _0, _1)
#define movups(_0, _1) INSTR_(movups, _0, _1)
#define movapd(_0, _1) INSTR_(movapd, _0, _1)
#define movupd(_0, _1) INSTR_(movupd, _0, _1)
#define movddup(_0, _1) INSTR_(movddup, _0, _1)
#define movlps(_0, _1) INSTR_(movlps, _0, _1)
#define movhps(_0, _1) INSTR_(movhps, _0, _1)
#define movlpd(_0, _1) INSTR_(movlpd, _0, _1)
#define movhpd(_0, _1) INSTR_(movhpd, _0, _1)

#define vmovddup(_0, _1) INSTR_(vmovddup, _0, _1)
#define vmovsldup(_0, _1) INSTR_(vmovsldup, _0, _1)
#define vmovshdup(_0, _1) INSTR_(vmovshdup, _0, _1)
#define vmovd(_0, _1) INSTR_(vmovd, _0, _1)
#define vmovq(_0, _1) INSTR_(vmovq, _0, _1)
#define vmovss(_0, _1) INSTR_(vmovss, _0, _1)
#define vmovsd(_0, _1) INSTR_(vmovsd, _0, _1)
#define vmovaps(_0, _1) INSTR_(vmovaps, _0, _1)
#define vmovups(_0, _1) INSTR_(vmovups, _0, _1)
#define vmovapd(_0, _1) INSTR_(vmovapd, _0, _1)
#define vmovupd(_0, _1) INSTR_(vmovupd, _0, _1)
#define vmovlps(...) INSTR_(vmovlps, __VA_ARGS__)
#define vmovhps(...) INSTR_(vmovhps, __VA_ARGS__)
#define vmovlpd(...) INSTR_(vmovlpd, __VA_ARGS__)
#define vmovhpd(...) INSTR_(vmovhpd, __VA_ARGS__)
#define vmovdqa(_0, _1) INSTR_(vmovdqa, _0, _1)
#define vmovdqa32(_0, _1) INSTR_(vmovdqa32, _0, _1)
#define vmovdqa64(_0, _1) INSTR_(vmovdqa64, _0, _1)
#define vbroadcastss(_0, _1) INSTR_(vbroadcastss, _0, _1)
#define vbroadcastsd(_0, _1) INSTR_(vbroadcastsd, _0, _1)
#define vpbroadcastd(_0, _1) INSTR_(vpbroadcastd, _0, _1)
#define vpbroadcastq(_0, _1) INSTR_(vpbroadcastq, _0, _1)
#define vbroadcastf128(_0, _1) INSTR_(vbroadcastf128, _0, _1)
#define vbroadcastf64x4(_0, _1) INSTR_(vbroadcastf64x4, _0, _1)
#define vgatherdps(...) INSTR_(vgatherdps, __VA_ARGS__)
#define vscatterdps(_0, _1) INSTR_(vscatterdps, _0, _1)
#define vgatherdpd(...) INSTR_(vgatherdpd, __VA_ARGS__)
#define vscatterdpd(_0, _1) INSTR_(vscatterdpd, _0, _1)
#define vgatherqps(...) INSTR_(vgatherqps, __VA_ARGS__)
#define vscatterqps(_0, _1) INSTR_(vscatterqps, _0, _1)
#define vgatherqpd(...) INSTR_(vgatherqpd, __VA_ARGS__)
#define vscatterqpd(_0, _1) INSTR_(vscatterqpd, _0, _1)

/*
 * Vector comparisons
 */
#define vpcmpeqb(_0, _1, _2) INSTR_(vpcmpeqb, _0, _1, _2)
#define vpcmpeqw(_0, _1, _2) INSTR_(vpcmpeqw, _0, _1, _2)
#define vpcmpeqd(_0, _1, _2) INSTR_(vpcmpeqd, _0, _1, _2)
#define maxps(_0, _1) INSTR_(maxps, _0, _1)
#define vmaxps(_0, _1, _2) INSTR_(vmaxps, _0, _1, _2)

/*
 * Vector integer math
 */
#define vpaddb(_0, _1, _2) INSTR_(vpaddb, _0, _1, _2)
#define vpaddw(_0, _1, _2) INSTR_(vpaddw, _0, _1, _2)
#define vpaddd(_0, _1, _2) INSTR_(vpaddd, _0, _1, _2)
#define vpaddq(_0, _1, _2) INSTR_(vpaddq, _0, _1, _2)

/*
 * Vector math
 */
#define addps(_0, _1) INSTR_(addps, _0, _1)
#define addpd(_0, _1) INSTR_(addpd, _0, _1)
#define subps(_0, _1) INSTR_(subps, _0, _1)
#define subpd(_0, _1) INSTR_(subpd, _0, _1)
#define mulps(_0, _1) INSTR_(mulps, _0, _1)
#define mulpd(_0, _1) INSTR_(mulpd, _0, _1)
#define divps(_0, _1) INSTR_(divps, _0, _1)
#define divpd(_0, _1) INSTR_(divpd, _0, _1)
#define xorps(_0, _1) INSTR_(xorps, _0, _1)
#define xorpd(_0, _1) INSTR_(xorpd, _0, _1)
#define rcpps(_0, _1) INSTR_(rcpps, _0, _1)

#define pmaddwd(_0, _1) INSTR_(pmaddwd, _0, _1)
#define paddd(_0, _1) INSTR_(paddd, _0, _1)
#define psubd(_0, _1) INSTR_(psubd, _0, _1)

#define ucomiss(_0, _1) INSTR_(ucomiss, _0, _1)
#define ucomisd(_0, _1) INSTR_(ucomisd, _0, _1)
#define comiss(_0, _1) INSTR_(comiss, _0, _1)
#define comisd(_0, _1) INSTR_(comisd, _0, _1)

#define vandps(_0, _1, _2) INSTR_(vandps, _0, _1, _2)
#define vorps(_0, _1, _2) INSTR_(vorps, _0, _1, _2)

#define vaddsubps(_0, _1, _2) INSTR_(vaddsubps, _0, _1, _2)
#define vaddsubpd(_0, _1, _2) INSTR_(vaddsubpd, _0, _1, _2)
#define vhaddpd(_0, _1, _2) INSTR_(vhaddpd, _0, _1, _2)
#define vhaddps(_0, _1, _2) INSTR_(vhaddps, _0, _1, _2)
#define vhsubpd(_0, _1, _2) INSTR_(vhsubpd, _0, _1, _2)
#define vhsubps(_0, _1, _2) INSTR_(vhsubps, _0, _1, _2)
#define vaddps(_0, _1, _2) INSTR_(vaddps, _0, _1, _2)
#define vaddpd(_0, _1, _2) INSTR_(vaddpd, _0, _1, _2)
#define vsubps(_0, _1, _2) INSTR_(vsubps, _0, _1, _2)
#define vsubpd(_0, _1, _2) INSTR_(vsubpd, _0, _1, _2)
#define vmulss(_0, _1, _2) INSTR_(vmulss, _0, _1, _2)
#define vmulsd(_0, _1, _2) INSTR_(vmulsd, _0, _1, _2)
#define vmulps(_0, _1, _2) INSTR_(vmulps, _0, _1, _2)
#define vmulpd(_0, _1, _2) INSTR_(vmulpd, _0, _1, _2)
#define vdivss(_0, _1, _2) INSTR_(vdivss, _0, _1, _2)
#define vdivsd(_0, _1, _2) INSTR_(vdivsd, _0, _1, _2)
#define vdivps(_0, _1, _2) INSTR_(vdivps, _0, _1, _2)
#define vdivpd(_0, _1, _2) INSTR_(vdivpd, _0, _1, _2)
#define vpmulld(_0, _1, _2) INSTR_(vpmulld, _0, _1, _2)
#define vpmullq(_0, _1, _2) INSTR_(vpmullq, _0, _1, _2)
#define vpaddd(_0, _1, _2) INSTR_(vpaddd, _0, _1, _2)
#define vpslld(_0, _1, _2) INSTR_(vpslld, _0, _1, _2)
#define vxorps(_0, _1, _2) INSTR_(vxorps, _0, _1, _2)
#define vxorpd(_0, _1, _2) INSTR_(vxorpd, _0, _1, _2)
#define vpxord(_0, _1, _2) INSTR_(vpxord, _0, _1, _2)
#define vrcpps(_0, _1) INSTR_(vrcpps, _0, _1)
#define vrcp14ps(_0, _1) INSTR_(vrcp14ps, _0, _1)

#define vpmaddwd(_0, _1, _2) INSTR_(vpmaddwd, _0, _1, _2)
#define vpmaddubsw(_0, _1, _2) INSTR_(vpmaddubsw, _0, _1, _2)
#define vpaddd(_0, _1, _2) INSTR_(vpaddd, _0, _1, _2)
#define vpsubd(_0, _1, _2) INSTR_(vpsubd, _0, _1, _2)

#define vucomiss(_0, _1) INSTR_(vucomiss, _0, _1)
#define vucomisd(_0, _1) INSTR_(vucomisd, _0, _1)
#define vcomiss(_0, _1) INSTR_(vcomiss, _0, _1)
#define vcomisd(_0, _1) INSTR_(vcomisd, _0, _1)

#define vfmadd132ss(_0, _1, _2) INSTR_(vfmadd132ss, _0, _1, _2)
#define vfmadd213ss(_0, _1, _2) INSTR_(vfmadd213ss, _0, _1, _2)
#define vfmadd231ss(_0, _1, _2) INSTR_(vfmadd231ss, _0, _1, _2)
#define vfmadd132sd(_0, _1, _2) INSTR_(vfmadd132sd, _0, _1, _2)
#define vfmadd213sd(_0, _1, _2) INSTR_(vfmadd213sd, _0, _1, _2)
#define vfmadd231sd(_0, _1, _2) INSTR_(vfmadd231sd, _0, _1, _2)
#define vfmadd132ps(_0, _1, _2) INSTR_(vfmadd132ps, _0, _1, _2)
#define vfmadd213ps(_0, _1, _2) INSTR_(vfmadd213ps, _0, _1, _2)
#define vfmadd231ps(_0, _1, _2) INSTR_(vfmadd231ps, _0, _1, _2)
#define vfmadd132pd(_0, _1, _2) INSTR_(vfmadd132pd, _0, _1, _2)
#define vfmadd213pd(_0, _1, _2) INSTR_(vfmadd213pd, _0, _1, _2)
#define vfmadd231pd(_0, _1, _2) INSTR_(vfmadd231pd, _0, _1, _2)

#define vfmsub132ss(_0, _1, _2) INSTR_(vfmsub132ss, _0, _1, _2)
#define vfmsub213ss(_0, _1, _2) INSTR_(vfmsub213ss, _0, _1, _2)
#define vfmsub231ss(_0, _1, _2) INSTR_(vfmsub231ss, _0, _1, _2)
#define vfmsub132sd(_0, _1, _2) INSTR_(vfmsub132sd, _0, _1, _2)
#define vfmsub213sd(_0, _1, _2) INSTR_(vfmsub213sd, _0, _1, _2)
#define vfmsub231sd(_0, _1, _2) INSTR_(vfmsub231sd, _0, _1, _2)
#define vfmsub132ps(_0, _1, _2) INSTR_(vfmsub132ps, _0, _1, _2)
#define vfmsub213ps(_0, _1, _2) INSTR_(vfmsub213ps, _0, _1, _2)
#define vfmsub231ps(_0, _1, _2) INSTR_(vfmsub231ps, _0, _1, _2)
#define vfmsub132pd(_0, _1, _2) INSTR_(vfmsub132pd, _0, _1, _2)
#define vfmsub213pd(_0, _1, _2) INSTR_(vfmsub213pd, _0, _1, _2)
#define vfmsub231pd(_0, _1, _2) INSTR_(vfmsub231pd, _0, _1, _2)

#define vfnmadd132ss(_0, _1, _2) INSTR_(vfnmadd132ss, _0, _1, _2)
#define vfnmadd213ss(_0, _1, _2) INSTR_(vfnmadd213ss, _0, _1, _2)
#define vfnmadd231ss(_0, _1, _2) INSTR_(vfnmadd231ss, _0, _1, _2)
#define vfnmadd132sd(_0, _1, _2) INSTR_(vfnmadd132sd, _0, _1, _2)
#define vfnmadd213sd(_0, _1, _2) INSTR_(vfnmadd213sd, _0, _1, _2)
#define vfnmadd231sd(_0, _1, _2) INSTR_(vfnmadd231sd, _0, _1, _2)
#define vfnmadd132ps(_0, _1, _2) INSTR_(vfnmadd132ps, _0, _1, _2)
#define vfnmadd213ps(_0, _1, _2) INSTR_(vfnmadd213ps, _0, _1, _2)
#define vfnmadd231ps(_0, _1, _2) INSTR_(vfnmadd231ps, _0, _1, _2)
#define vfnmadd132pd(_0, _1, _2) INSTR_(vfnmadd132pd, _0, _1, _2)
#define vfnmadd213pd(_0, _1, _2) INSTR_(vfnmadd213pd, _0, _1, _2)
#define vfnmadd231pd(_0, _1, _2) INSTR_(vfnmadd231pd, _0, _1, _2)

#define vfnmsub132ss(_0, _1, _2) INSTR_(vfnmsub132ss, _0, _1, _2)
#define vfnmsub213ss(_0, _1, _2) INSTR_(vfnmsub213ss, _0, _1, _2)
#define vfnmsub231ss(_0, _1, _2) INSTR_(vfnmsub231ss, _0, _1, _2)
#define vfnmsub132sd(_0, _1, _2) INSTR_(vfnmsub132sd, _0, _1, _2)
#define vfnmsub213sd(_0, _1, _2) INSTR_(vfnmsub213sd, _0, _1, _2)
#define vfnmsub231sd(_0, _1, _2) INSTR_(vfnmsub231sd, _0, _1, _2)
#define vfnmsub132ps(_0, _1, _2) INSTR_(vfnmsub132ps, _0, _1, _2)
#define vfnmsub213ps(_0, _1, _2) INSTR_(vfnmsub213ps, _0, _1, _2)
#define vfnmsub231ps(_0, _1, _2) INSTR_(vfnmsub231ps, _0, _1, _2)
#define vfnmsub132pd(_0, _1, _2) INSTR_(vfnmsub132pd, _0, _1, _2)
#define vfnmsub213pd(_0, _1, _2) INSTR_(vfnmsub213pd, _0, _1, _2)
#define vfnmsub231pd(_0, _1, _2) INSTR_(vfnmsub231pd, _0, _1, _2)

#define vfmaddsub132ss(_0, _1, _2) INSTR_(vfmaddsub132ss, _0, _1, _2)
#define vfmaddsub213ss(_0, _1, _2) INSTR_(vfmaddsub213ss, _0, _1, _2)
#define vfmaddsub231ss(_0, _1, _2) INSTR_(vfmaddsub231ss, _0, _1, _2)
#define vfmaddsub132sd(_0, _1, _2) INSTR_(vfmaddsub132sd, _0, _1, _2)
#define vfmaddsub213sd(_0, _1, _2) INSTR_(vfmaddsub213sd, _0, _1, _2)
#define vfmaddsub231sd(_0, _1, _2) INSTR_(vfmaddsub231sd, _0, _1, _2)
#define vfmaddsub132ps(_0, _1, _2) INSTR_(vfmaddsub132ps, _0, _1, _2)
#define vfmaddsub213ps(_0, _1, _2) INSTR_(vfmaddsub213ps, _0, _1, _2)
#define vfmaddsub231ps(_0, _1, _2) INSTR_(vfmaddsub231ps, _0, _1, _2)
#define vfmaddsub132pd(_0, _1, _2) INSTR_(vfmaddsub132pd, _0, _1, _2)
#define vfmaddsub213pd(_0, _1, _2) INSTR_(vfmaddsub213pd, _0, _1, _2)
#define vfmaddsub231pd(_0, _1, _2) INSTR_(vfmaddsub231pd, _0, _1, _2)
#define vfmsubadd132ss(_0, _1, _2) INSTR_(vfmsubadd132ss, _0, _1, _2)
#define vfmsubadd213ss(_0, _1, _2) INSTR_(vfmsubadd213ss, _0, _1, _2)
#define vfmsubadd231ss(_0, _1, _2) INSTR_(vfmsubadd231ss, _0, _1, _2)
#define vfmsubadd132sd(_0, _1, _2) INSTR_(vfmsubadd132sd, _0, _1, _2)
#define vfmsubadd213sd(_0, _1, _2) INSTR_(vfmsubadd213sd, _0, _1, _2)
#define vfmsubadd231sd(_0, _1, _2) INSTR_(vfmsubadd231sd, _0, _1, _2)
#define vfmsubadd132ps(_0, _1, _2) INSTR_(vfmsubadd132ps, _0, _1, _2)
#define vfmsubadd213ps(_0, _1, _2) INSTR_(vfmsubadd213ps, _0, _1, _2)
#define vfmsubadd231ps(_0, _1, _2) INSTR_(vfmsubadd231ps, _0, _1, _2)
#define vfmsubadd132pd(_0, _1, _2) INSTR_(vfmsubadd132pd, _0, _1, _2)
#define vfmsubadd213pd(_0, _1, _2) INSTR_(vfmsubadd213pd, _0, _1, _2)
#define vfmsubadd231pd(_0, _1, _2) INSTR_(vfmsubadd231pd, _0, _1, _2)

#define vfmaddss(_0, _1, _2, _3) INSTR_(vfmaddss, _0, _1, _2, _3)
#define vfmaddsd(_0, _1, _2, _3) INSTR_(vfmaddsd, _0, _1, _2, _3)
#define vfmaddps(_0, _1, _2, _3) INSTR_(vfmaddps, _0, _1, _2, _3)
#define vfmaddpd(_0, _1, _2, _3) INSTR_(vfmaddpd, _0, _1, _2, _3)
#define vfmsubss(_0, _1, _2, _3) INSTR_(vfmsubss, _0, _1, _2, _3)
#define vfmsubsd(_0, _1, _2, _3) INSTR_(vfmsubsd, _0, _1, _2, _3)
#define vfmsubps(_0, _1, _2, _3) INSTR_(vfmsubps, _0, _1, _2, _3)
#define vfmsubpd(_0, _1, _2, _3) INSTR_(vfmsubpd, _0, _1, _2, _3)
#define vfnmaddss(_0, _1, _2, _3) INSTR_(vfnmaddss, _0, _1, _2, _3)
#define vfnmaddsd(_0, _1, _2, _3) INSTR_(vfnmaddsd, _0, _1, _2, _3)
#define vfnmaddps(_0, _1, _2, _3) INSTR_(vfnmaddps, _0, _1, _2, _3)
#define vfnmaddpd(_0, _1, _2, _3) INSTR_(vfnmaddpd, _0, _1, _2, _3)
#define vfnmsubss(_0, _1, _2, _3) INSTR_(vfnmsubss, _0, _1, _2, _3)
#define vfnmsubsd(_0, _1, _2, _3) INSTR_(vfnmsubsd, _0, _1, _2, _3)
#define vfnmsubps(_0, _1, _2, _3) INSTR_(vfnmsubps, _0, _1, _2, _3)
#define vfnmsubpd(_0, _1, _2, _3) INSTR_(vfnmsubpd, _0, _1, _2, _3)

#define vfmaddsubss(_0, _1, _2, _3) INSTR_(vfmaddsubss, _0, _1, _2, _3)
#define vfmaddsubsd(_0, _1, _2, _3) INSTR_(vfmaddsubsd, _0, _1, _2, _3)
#define vfmaddsubps(_0, _1, _2, _3) INSTR_(vfmaddsubps, _0, _1, _2, _3)
#define vfmaddsubpd(_0, _1, _2, _3) INSTR_(vfmaddsubpd, _0, _1, _2, _3)
#define vfmsubaddss(_0, _1, _2, _3) INSTR_(vfmsubaddss, _0, _1, _2, _3)
#define vfmsubaddsd(_0, _1, _2, _3) INSTR_(vfmsubaddsd, _0, _1, _2, _3)
#define vfmsubaddps(_0, _1, _2, _3) INSTR_(vfmsubaddps, _0, _1, _2, _3)
#define vfmsubaddpd(_0, _1, _2, _3) INSTR_(vfmsubaddpd, _0, _1, _2, _3)

#define v4fmaddss(_0, _1, _2) INSTR_(v4fmaddss, _0, _1, _2)
#define v4fmaddps(_0, _1, _2) INSTR_(v4fmaddps, _0, _1, _2)
#define v4fnmaddss(_0, _1, _2) INSTR_(v4fnmaddss, _0, _1, _2)
#define v4fnmaddps(_0, _1, _2) INSTR_(v4fnmaddps, _0, _1, _2)

/*
 * Conversions
 */
#define cvtss2sd(_0, _1) INSTR_(cvtss2sd, _0, _1)
#define cvtsd2ss(_0, _1) INSTR_(cvtsd2ss, _0, _1)
#define cvtps2pd(_0, _1) INSTR_(cvtps2pd, _0, _1)
#define cvtpd2ps(_0, _1) INSTR_(cvtpd2ps, _0, _1)
#define cvtps2dq(_0, _1) INSTR_(cvtps2dq, _0, _1)

#define vcvtss2sd(_0, _1) INSTR_(vcvtss2sd, _0, _1)
#define vcvtsd2ss(_0, _1) INSTR_(vcvtsd2ss, _0, _1)
#define vcvtps2pd(_0, _1) INSTR_(vcvtps2pd, _0, _1)
#define vcvtpd2ps(_0, _1) INSTR_(vcvtpd2ps, _0, _1)
#define vcvtps2dq(_0, _1) INSTR_(vcvtps2dq, _0, _1)
#define vcvtdq2ps(_0, _1) INSTR_(vcvtdq2ps, _0, _1)

#define vcvtph2ps(_0, _1) INSTR_(vcvtph2ps, _0, _1)
#define vcvtps2ph(_0, _1, _2) INSTR_(vcvtps2ph, _0, _1, _2)



/*
 * Vector shuffles
 */
#define pshufd(_0, _1, _2) INSTR_(pshufd, _0, _1, _2)
#define shufps(_0, _1, _2) INSTR_(shufps, _0, _1, _2)
#define shufpd(_0, _1, _2) INSTR_(shufpd, _0, _1, _2)
#define unpcklps(_0, _1) INSTR_(unpcklps, _0, _1)
#define unpckhps(_0, _1) INSTR_(unpckhps, _0, _1)
#define unpcklpd(_0, _1) INSTR_(unpcklpd, _0, _1)
#define unpckhpd(_0, _1) INSTR_(unpckhpd, _0, _1)
#define movlhps(_0, _1) INSTR_(movlhps, _0, _1)
#define movhlps(_0, _1) INSTR_(movhlps, _0, _1)

#define vshufps(_0, _1, _2, _3) INSTR_(vshufps, _0, _1, _2, _3)
#define vshufpd(_0, _1, _2, _3) INSTR_(vshufpd, _0, _1, _2, _3)
#define vpermilps(_0, _1, _2) INSTR_(vpermilps, _0, _1, _2)
#define vpermilpd(_0, _1, _2) INSTR_(vpermilpd, _0, _1, _2)
#define vperm2f128(_0, _1, _2, _3) INSTR_(vperm2f128, _0, _1, _2, _3)
#define vpermpd(_0, _1, _2) INSTR_(vpermpd, _0, _1, _2)
#define vunpcklps(_0, _1, _2) INSTR_(vunpcklps, _0, _1, _2)
#define vunpckhps(_0, _1, _2) INSTR_(vunpckhps, _0, _1, _2)
#define vunpcklpd(_0, _1, _2) INSTR_(vunpcklpd, _0, _1, _2)
#define vunpckhpd(_0, _1, _2) INSTR_(vunpckhpd, _0, _1, _2)
#define vshuff32x4(_0, _1, _2, _3) INSTR_(vshuff32x4, _0, _1, _2, _3)
#define vshuff64x2(_0, _1, _2, _3) INSTR_(vshuff64x2, _0, _1, _2, _3)
#define vinsertf128(_0, _1, _2, _3) INSTR_(vinsertf128, _0, _1, _2, _3)
#define vinsertf32x4(_0, _1, _2, _3) INSTR_(vinsertf32x4, _0, _1, _2, _3)
#define vinsertf32x8(_0, _1, _2, _3) INSTR_(vinsertf32x8, _0, _1, _2, _3)
#define vinsertf64x2(_0, _1, _2, _3) INSTR_(vinsertf64x2, _0, _1, _2, _3)
#define vinsertf64x4(_0, _1, _2, _3) INSTR_(vinsertf64x4, _0, _1, _2, _3)
#define vextractf128(_0, _1, _2) INSTR_(vextractf128, _0, _1, _2)
#define vextractf32x4(_0, _1, _2) INSTR_(vextractf32x4, _0, _1, _2)
#define vextractf32x8(_0, _1, _2) INSTR_(vextractf32x8, _0, _1, _2)
#define vextractf64x2(_0, _1, _2) INSTR_(vextractf64x2, _0, _1, _2)
#define vextractf64x4(_0, _1, _2) INSTR_(vextractf64x4, _0, _1, _2)
#define vblendps(_0, _1, _2, _3) INSTR_(vblendps, _0, _1, _2, _3)
#define vblendpd(_0, _1, _2, _3) INSTR_(vblendpd, _0, _1, _2, _3)
#define vblendmps(_0, _1, _2) INSTR_(vblendmps, _0, _1, _2)
#define vblendmpd(_0, _1, _2) INSTR_(vblendmpd, _0, _1, _2)
#define vmovlhps(_0, _1, _2) INSTR_(vmovlhps, _0, _1, _2)
#define vmovhlps(_0, _1, _2) INSTR_(vmovhlps, _0, _1, _2)

/*
 * Prefetches
 */

#define prefetch(_0, _1) INSTR_(prefetcht##_0, _1)
#define prefetchw(_0) INSTR_(prefetchw, _0)
#define prefetchw0(_0) INSTR_(prefetchwt1, _0)

/*
 * Other
 */

#define rdtsc() INSTR_(rdtsc)
#define vzeroall() INSTR_(vzeroall)
#define vzeroupper() INSTR_(vzeroupper)

#endif /* BACKEND_CPU_ASSEMBLY_MACROS_HPP_ */
