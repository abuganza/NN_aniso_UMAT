      SUBROUTINE USDFLD(FIELD,STATEV,PNEWDT,DIRECT,T,CELENT,
     1 TIME,DTIME,CMNAME,ORNAME,NFIELD,NSTATV,NOEL,NPT,LAYER,
     2 KSPT,KSTEP,KINC,NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,LACCFLA)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME,ORNAME
      CHARACTER*3  FLGRAY(15)
      DIMENSION FIELD(NFIELD),STATEV(NSTATV),DIRECT(3,3),
     1 T(3,3),TIME(2)
      DIMENSION ARRAY(15),JARRAY(15),JMAC(*),JMATYP(*),COORD(*)

      STATEV(2) = DIRECT(1,1)
      STATEV(3) = DIRECT(2,1)
      STATEV(4) = DIRECT(3,1)
      STATEV(5) = DIRECT(1,2)
      STATEV(6) = DIRECT(2,2)
      STATEV(7) = DIRECT(3,2)
      STATEV(8) = DIRECT(1,3)
      STATEV(9) = DIRECT(2,3)
      STATEV(10) = DIRECT(3,3)
C      user coding to define FIELD and, if necessary, STATEV and PNEWDT
      FIELD(1) = 0.5

C      OPEN(16,file='C:\temp\USD_OUT.txt',status='old',access='append')
C      WRITE(16,101) STATEV(1), STATEV(2), STATEV(3), STATEV(4),
C     1 STATEV(5), STATEV(6), STATEV(7), STATEV(8), STATEV(9), STATEV(10)
C101   FORMAT(' ',10G10.2)
C      CLOSE(16)

      RETURN
      END SUBROUTINE
c...  ------------------------------------------------------------------
      subroutine sdvini(statev,coords,nstatv,ncrds,noel,npt,layer,kspt)
c...  ------------------------------------------------------------------
      include 'aba_param.inc'


      dimension statev(nstatv)

      statev(1)=1.0d0    

      return
      end



c...  ------------------------------------------------------------------
      subroutine umat(stress,statev,ddsdde,sse,spd,scd,
     #rpl,ddsddt,drplde,drpldt,
     #stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname,
     #ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt,
     #celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,kstep,kinc)
c...  ------------------------------------------------------------------
      include 'aba_param.inc'

      character*80 cmname
      dimension stress(ntens),statev(nstatv),
     #ddsdde(ntens,ntens),ddsddt(ntens),drplde(ntens),
     #stran(ntens),dstran(ntens),time(2),predef(1),dpred(1),
     #props(nprops),coords(3),drot(3,3),dfgrd0(3,3),dfgrd1(3,3)

      call umat_NN(stress,statev,ddsdde,sse,
     #                       time,dtime,coords,props,dfgrd1,
     #                       ntens,ndi,nshr,nstatv,nprops,
     #                       noel,npt,kstep,kinc)

      return
      end

c...  ------------------------------------------------------------------
      subroutine umat_NN(stress,statev,ddsdde,sse,
     #                             time,dtime,coords,props,dfgrd1,
     #                             ntens,ndi,nshr,nstatv,nprops,
     #                             noel,npt,kstep,kinc)
c...  ------------------------------------------------------------------

c...  ------------------------------------------------------------------

      implicit none

c...  variables to be defined
      real*8  stress(ntens), ddsdde(ntens,ntens), statev(nstatv), sse

c...  variables passed in for information
      real*8  time(2), dtime, coords(3), props(nprops), dfgrd1(3,3)
      integer ntens, ndi, nshr, nstatv, nprops, noel, npt, kstep, kinc

c...  local variables (mostly mechanics part)
      real*8 Fhat(3,3), detf, C(3,3), Cinv(3,3), detC, Chat(3,3)
      real*8 trChat2
      real*8 sigma(3,3), S(3,3), Shat(3,3), Siso(3,3), trS
      real*8 p, Psivol, dpdJ, Kvol, ptilde
      real*8 I1hat, I2hat, I4ahat, I4shat
      real*8 Psi1, Psi11, Psi12, Psi14a, Psi14s
      real*8 Psi2, Psi21, Psi22, Psi24a, Psi24s
      real*8 Psi4a, Psi4a1, Psi4a2, Psi4a4a, Psi4a4s
      real*8 Psi4s, Psi4s1, Psi4s2, Psi4s4a, Psi4s4s
      real*8 delta1, delta2, delta3, delta4, delta5, delta6, delta7
      real*8 delta8, delta9, delta10, delta11
      real*8 IIII(3,3,3,3), CChat(3,3,3,3), CCvol(3,3,3,3), CCiso(3,3,3,3)
      real*8 CC_ref(3,3,3,3), cc_def(3,3,3,3), PP1(3,3,3,3), PP2(3,3,3,3)
      real*8 dShatdC(3,3,3,3)
      real*8 kron(3,3), A0(3,3), S0(3,3), a0vector(3), s0vector(3)

c...  some auxiliar variables, tensors 
      integer i, j, k, l, II, JJ, Itoi(6), Itoj(6)
      integer i2, j2, k2, l2

c...  material properties, read in the weights and biases 
      integer weight_count, ind, bias_count, n_input, nlayers 
c...  values needed for normalization of I1, I2, in our approach
      real*8 I1mean,I1var,I2mean,I2var, I4amean, I4avar, I4smean
      real*8 I4svar, Psimean, Psivar
c...  nlayers = props(1)
c...  n_input = props(2)
      integer n_neuronsperlayer(props(1)) 
      real*8 input_param(props(2))
c...  weight_count = props(3)
c...  bias_count = props(4)
      real*8 ALLweights(props(3)), ALLbiases(props(4)) 
      integer activtypes(props(1)-1)
      real*8 activout, gradactivout
c...      real*8 output_vector(bias_count+n_input+2)
c...      real*8 output_gradient((n_input+2)*(n_input+2+bias_count))
      real*8 output_vector(props(4)+props(2)+2)
      real*8 output_grad((props(2)+2)*(props(2)+2+props(4)) )
      integer io1, iw1, ig1, io2, iw2, ig2, ib1


c...  initialize material parameters
      nlayers   = props(1)    ! number of layers of the NN, including input
      do i = 1,nlayers 
        n_neuronsperlayer(i)    = props(i+4)
      end do
c...  read the normalizing constants for I1, I2
      I1mean  = props(4+nlayers+1)
      I1var   = props(4+nlayers+2)
      I2mean  = props(4+nlayers+3)
      I2var   = props(4+nlayers+4)
      I4amean = props(4+nlayers+5)
      I4avar  = props(4+nlayers+6)
      I4smean = props(4+nlayers+7)
      I4svar  = props(4+nlayers+8)
      Psimean = props(4+nlayers+9)
      Psivar  = props(4+nlayers+10)

      n_input = props(2)
      
c...  read in weights and biases
      weight_count = props(3)
      bias_count = props(4)
      ind = 1
      do i=1,nlayers-1
        do j=1,n_neuronsperlayer(i)
          do k=1,n_neuronsperlayer(i+1)
            ALLweights(ind) = props(4+nlayers+10+ind)
            ind = ind+1
          end do
        end do
      end do
      ind = 1
      do i=2,nlayers
        do j=1,n_neuronsperlayer(i)
          ALLbiases(ind) = props(4+nlayers+10+weight_count+ind)
          ind = ind + 1
        end do
      end do
      ind = 1
      do i=2,nlayers
        activtypes(ind)=int(props(4+nlayers+10+weight_count+bias_count+ind))
        ind = ind+1
      end do

      Kvol = props(4+nlayers+10+weight_count+bias_count+nlayers)
      do i=1,3
        a0vector(i) = props(4+nlayers+10+weight_count+bias_count+nlayers+i)
      end do
      do i=1,3
        s0vector(i) = props(4+nlayers+10+weight_count+bias_count+nlayers+3+i)
      end do

c...  calculate determinant of deformation gradient
      detf = det(dfgrd1)

c...      C = F^T*F
c...      b = F*F^T -> full notation [[b11, b12, b13],[b12,b22,b23],[b13,b23,b33]]
c...      b = [b11,b22,b33,b12,b13,b23] -> voigt notation

c...  calculate right cauchy-green deformation tensor C

      C = matmul(transpose(dfgrd1), dfgrd1)

      Chat = C*detf**(-2./3.)

      detC = det(C)

      Cinv(1,1) = (+C(2,2)*C(3,3) - C(2,3)*C(3,2))/detC
      Cinv(1,2) = (-C(1,2)*C(3,3) + C(1,3)*C(3,2))/detC
      Cinv(1,3) = (+C(1,2)*C(2,3) - C(1,3)*C(2,2))/detC
      Cinv(2,1) = (-C(2,1)*C(3,3) + C(2,3)*C(3,1))/detC
      Cinv(2,2) = (+C(1,1)*C(3,3) - C(1,3)*C(3,1))/detC
      Cinv(2,3) = (-C(1,1)*C(2,3) + C(1,3)*C(2,1))/detC
      Cinv(3,1) = (+C(2,1)*C(3,2) - C(2,2)*C(3,1))/detC
      Cinv(3,2) = (-C(1,1)*C(3,2) + C(1,2)*C(3,1))/detC
      Cinv(3,3) = (+C(1,1)*C(2,2) - C(1,2)*C(2,1))/detC

c...  get the isochoric split 
      do i=1,3
        do j=1,3
          Fhat(i,j) = dfgrd1(i,j)*detf**(-1./3.)
        end do
      end do

c... get the invariants of Chat
      I1hat = Chat(1,1)+Chat(2,2)+Chat(3,3)
      trChat2 =   Chat(1,1)**2 +   Chat(2,2)**2 +   Chat(3,3)**2
     #         +2*Chat(1,2)**2 + 2*Chat(1,3)**2 + 2*Chat(2,3)**2
      I2hat = 0.5*(I1hat*I1hat-trChat2)
      I4ahat = 0
      I4shat = 0
      do i=1,3
        do j=1,3
          I4ahat = I4ahat + a0vector(i)*Chat(i,j)*a0vector(j)
          I4shat = I4shat + s0vector(i)*Chat(i,j)*s0vector(j)
        end do
      end do
c...  evaluate the NN and derivatives 
c...  fill out the input vector
      output_vector = 0
      output_grad = 0
c...  CAREFUL!!!!!!! 
c...  Normalizing the input 
c...  If your NN was trained with I1, I2 directly then you can pass 0 for the mean and 1 for the variance 
      output_vector(1) = (I1hat-I1mean)/I1var
      output_vector(2) = (I2hat-I2mean)/I2var
      output_vector(3) = (I4ahat-I4amean)/I4avar
      output_vector(4) = (I4shat-I4smean)/I4svar
      output_grad(0*n_input+1) = 1
      output_grad(1*n_input+2) = 1
      output_grad(2*n_input+3) = 1
      output_grad(3*n_input+4) = 1
      io1 = 0
      iw1 = 0
      ig1 = 0
      ib1 = 0
      do i =1,nlayers-1
c...    Beginning and end of the chunk in output vector to be used as input
        io2 = io1 + n_neuronsperlayer(i) 
c...    Beginning and end of the chunk in weight array defining matrix 
        iw2 = iw1 + n_neuronsperlayer(i)*n_neuronsperlayer(i+1) 
c...    Beginning and end of the chunk for grad outputs 
        ig2 = ig1 + n_neuronsperlayer(i)*n_input
c...    do the matrix vector product and store in output chunk 
        do k = 1,n_neuronsperlayer(i+1)
          do j = 1,n_neuronsperlayer(i) 
c...        Matrix*vector + bias 
            output_vector(io2+k) = output_vector(io2+k) + ALLweights(iw1+(j-1)* 
     #                n_neuronsperlayer(i+1)+k)*output_vector(io1+j)
c...        Matrix*Matrix for jacobian 
            do l = 1,n_input
              output_grad(ig2+(k-1)*n_input+l) = output_grad(ig2+(k-1)*n_input+l ) +
     #                ALLweights(iw1+(j-1)*n_neuronsperlayer(i+1)+k)
     #                *output_grad(ig1+(j-1)*n_input+l)
            end do
          end do
          activout = output_vector(io2+k) + ALLbiases(ib1+k)
          gradactivout = activout
          call activation(activout,activtypes(i))
          output_vector(io2+k) = activout
          call grad_activation(gradactivout,activtypes(i))
          do l = 1,n_input
            output_grad(ig2+(k-1)*n_input+l) = gradactivout*output_grad(ig2+(k-1)*n_input+l)
          end do
        end do
        io1 = io2 
        iw1 = iw2
        ig1 = ig2
        ib1 = ib1 + n_neuronsperlayer(i+1)
      end do

      io2 = n_neuronsperlayer(i)
c... here we should have first derivatives Psi1, Psi2
c... below the index of bias_count+1 corresponds to the first output, which is Psi
      Psi1  = output_vector(bias_count+1)
      Psi2  = output_vector(bias_count+2)
      Psi4a = output_vector(bias_count+3)
      Psi4s = output_vector(bias_count+4)
c... and second derivatives Psi11,Psi12,Psi21,Psi22
      Psi11   = output_grad((0+bias_count)*n_input+1)
      Psi12   = output_grad((0+bias_count)*n_input+2)
      Psi14a  = output_grad((0+bias_count)*n_input+3)
      Psi14s  = output_grad((0+bias_count)*n_input+4)
      Psi21   = output_grad((1+bias_count)*n_input+1)
      Psi22   = output_grad((1+bias_count)*n_input+2)
      Psi24a  = output_grad((1+bias_count)*n_input+3)
      Psi24s  = output_grad((1+bias_count)*n_input+4)
      Psi4a1  = output_grad((2+bias_count)*n_input+1)
      Psi4a2  = output_grad((2+bias_count)*n_input+2)
      Psi4a4a = output_grad((2+bias_count)*n_input+3)
      Psi4a4s = output_grad((2+bias_count)*n_input+4)
      Psi4s1  = output_grad((3+bias_count)*n_input+1)
      Psi4s2  = output_grad((3+bias_count)*n_input+2)
      Psi4s4a = output_grad((3+bias_count)*n_input+3)
      Psi4s4s = output_grad((3+bias_count)*n_input+4)
c... "Unnormalize"?
      Psi1  = Psi1  *Psivar/I1var
      Psi2  = Psi2  *Psivar/I2var
      Psi4a = Psi4a *Psivar/I4avar
      Psi4s = Psi4s *Psivar/I4svar
      Psi11   = Psi11   *Psivar/I1var**2
      Psi12   = Psi12   *Psivar/I1var/I2var
      Psi14a  = Psi14a  *Psivar/I1var/I4avar
      Psi14s  = Psi14s  *Psivar/I1var/I4svar
      Psi21   = Psi21   *Psivar/I2var/I1var
      Psi22   = Psi22   *Psivar/I2var**2
      Psi24a  = Psi24a  *Psivar/I2var/I4avar
      Psi24s  = Psi24s  *Psivar/I2var/I4svar
      Psi4a1  = Psi4a1  *Psivar/I4avar/I1var
      Psi4a2  = Psi4a2  *Psivar/I4avar/I2var
      Psi4a4a = Psi4a4a *Psivar/I4avar**2
      Psi4a4s = Psi4a4s *Psivar/I4avar/I4svar
      Psi4s1  = Psi4s1  *Psivar/I4svar/I1var
      Psi4s2  = Psi4s2  *Psivar/I4svar/I2var
      Psi4s4a = Psi4s4a *Psivar/I4svar/I4avar
      Psi4s4s = Psi4s4s *Psivar/I4svar**2

c...  Kronecker delta. Note that kronecker delta is also
c...  2nd order identity.
      kron = 0
      kron(1,1) = 1.0
      kron(2,2) = 1.0
      kron(3,3) = 1.0

c...  Fictitious 2nd Piola Kirchoff Stress, Shat
      do i=1,3
        do j=1,3
          A0(i,j) = a0vector(i)*a0vector(j)
          S0(i,j) = s0vector(i)*s0vector(j)
          Shat(i,j) = 2*Psi1*kron(i,j) + 2*Psi2*(I1hat*kron(i,j) -
     #                Chat(i,j)) + 2*Psi4a*A0(i,j) + 2*Psi4s*S0(i,j)
        end do
      end do
c...  Pressure, p = dPsi_vol/dJ. Assume the following form for it for now: p = (J-1)^2 + log(J)
      Psivol = Kvol*(detf-1)**2
      p = 2*Kvol*(detf-1)
      dpdJ = 2*Kvol
      ptilde = p + detf*dpdJ
c...  4th order projection tensor, PP
      do i=1,3
        do j=1,3
          Siso(i,j) = 0
          do k=1,3
            do l=1,3
c...          previously I was using IIII(i,j,k,l) = kron(i,k)*kron(j,l) following Holzapfel 2000 page 229
c...          But I think that is a mistake. In reality you should use the symmetric identity.
              IIII(i,j,k,l) = 0.5*(kron(i,k)*kron(j,l) + kron(i,l)*kron(j,k))
c              IIII(i,j,k,l) = kron(i,k)*kron(j,l)
              PP1(i,j,k,l) = IIII(i,j,k,l) - 1./3.*Cinv(i,j)*C(k,l)
              PP2(i,j,k,l) =  0.5*(Cinv(i,k)*Cinv(j,l) + Cinv(i,l)*Cinv(j,k))
     #                           -1./3.*Cinv(i,j)*Cinv(k,l)
              Siso(i,j) = Siso(i,j) + detf**(-2./3.)*PP1(i,j,k,l)*Shat(k,l)
            end do
          end do
          S(i,j) = Siso(i,j) + detf*p*Cinv(i,j)
        end do
      end do
      trS = Shat(1,1) + Shat(2,2) + Shat(3,3)
      do i=1,3
        do j=1,3
          sigma(i,j) = 0
            do k=1,3
              do l=1,3
                sigma(i,j) = sigma(i,j) +
     #                          dfgrd1(i,k)*S(k,l)*dfgrd1(j,l)
              end do
            end do
          sigma(i,j) = sigma(i,j)/detf
        end do
      end do
      stress(1) = sigma(1,1)
      stress(2) = sigma(2,2)
      stress(3) = sigma(3,3)
      stress(4) = sigma(1,2)
      stress(5) = sigma(1,3)
      stress(6) = sigma(2,3)
c...  ==================================DEBUGGING==================================

c      if (noel.eq.500.and.npt.eq.1) then
c        write(7,*) "***************************************************"
c        write(7,*) "deformation gradient"
c        write(7,*) dfgrd1
c        write(7,*) "J"
c        write(7,*) detf
c        write(7,*) "C"
c        write(7,*) C
c        write(7,*) "Chat"
c        write(7,*) Chat
c        write(7,*) "I1hat, I2hat, I4ahat, I4shat"
c        write(7,*) I1hat, I2hat, I4ahat, I4shat
c        write(7,*) "S^hat"
c        write(7,*) Shat
c        write(7,*) "S_iso"
c        write(7,*) Siso
c        write(7,*) "S"
c        write(7,*) S
c        write(7,*) "sigma"
c        write(7,*) sigma
c        write(7,*) "Psi24a"
c        write(7,*) Psi24a
c      end if
c...  tangent in voigt notation (see holland abaqus documentation)
c...  The NN computes the two derivatives Psi12 and Psi21 independently, there
c...  is no guarantee that they are the same... this should be enforced during NN training
c...  Psi12 with the average of the two 
      Psi12   = 0.5*(Psi12   + Psi21  )
      Psi14a  = 0.5*(Psi14a  + Psi4a1 )
      Psi24a  = 0.5*(Psi24a  + Psi4a2 )
      Psi14s  = 0.5*(Psi14s  + Psi4s1 )
      Psi24s  = 0.5*(Psi24s  + Psi4s2 )
      Psi4a4s = 0.5*(Psi4a4s + Psi4s4a)

      delta1 =  2*(Psi2+Psi11+2*(I1hat*Psi12)+I1hat*I1hat*Psi22)
      delta2 = -2*(Psi12+I1hat*Psi22)
      delta3 =  2*(Psi14a + I1hat*Psi24a)
      delta4 =  2*(Psi14s + I1hat*Psi24s)
      delta5 = -2*Psi24a
      delta6 = -2*Psi24s
      delta7 =  2*Psi4a4s
      delta8 =  2*Psi4a4a
      delta9 =  2*Psi4s4s
      delta10 = 2*Psi22
      delta11 =-2*Psi2

      Itoi = (/ 1, 2, 3, 1, 1, 2 /)
      Itoj = (/ 1, 2, 3, 2, 3, 3 /)
c...  Fill part of the tangent in voigt notation
c...  1->11, 2->22, 3->33, 4->12, 5->13, 6->23
c...  Itoi = [1,2,3,1,1,2] (definition above)
c...  Itoj = [1,2,3,2,3,3] (definition above)
      do i=1,3
        do j=1,3
          do k=1,3
            do l=1,3
              CChat(i,j,k,l) = delta1*(kron(i,j)*kron(k,l)                      ) +
     #                         delta2*(Chat(i,j)*kron(k,l) + kron(i,j)*Chat(k,l)) +
     #                         delta3*(A0(i,j)*kron(k,l)   + kron(i,j)*A0(k,l)  ) +
     #                         delta4*(S0(i,j)*kron(k,l)   + kron(i,j)*S0(k,l)  ) +
     #                         delta5*(A0(i,j)*Chat(k,l)   + Chat(i,j)*A0(k,l)  ) +
     #                         delta6*(S0(i,j)*Chat(k,l)   + Chat(i,j)*S0(k,l)  ) +
     #                         delta7*(A0(i,j)*S0(k,l)     + S0(i,j)*A0(k,l)    ) +
     #                         delta8*(A0(i,j)*A0(k,l)                          ) +
     #                         delta9*(S0(i,j)*S0(k,l)                          ) +
     #                         delta10*(Chat(i,j)*Chat(k,l)                     ) +
     #                         delta11*IIII(i,j,k,l)
c...          here I include the 1/J^(-4/3) term inside CChat rather than multiplying it later.
              CChat(i,j,k,l) = CChat(i,j,k,l)*2.*detf**(-4./3.)
            end do
          end do
        end do
      end do

      do i=1,3
        do j=1,3
          do k=1,3
            do l=1,3
              CCvol(i,j,k,l) =  detf*ptilde*Cinv(i,j)*Cinv(k,l)
     #                         -detf*p*(Cinv(i,k)*Cinv(j,l) + Cinv(i,l)*Cinv(j,k))
              CCiso(i,j,k,l) = -2./3.*Siso(i,j)*Cinv(k,l) -2./3.*Cinv(i,j)*Siso(k,l)
     #                         +2./3.*detf**(-2./3.)*trS*PP2(i,j,k,l)
c...   The following 4 loops are for the 2 double contractions between PP, CChat and PP^T
              dShatdC(i,j,k,l) = 0
              do i2 = 1,3
                do j2 = 1,3
c...              Debugging: print dShat/dC for comparison
                  dShatdC(i,j,k,l) = dShatdC(i,j,k,l) + 0.5*detf**(2./3.)*CChat(i,j,i2,j2)*PP1(k,l,i2,j2)
                  do k2 = 1,3
                    do l2 = 1,3
                      CCiso(i,j,k,l) = CCiso(i,j,k,l) +
     #                                 PP1(i,j,i2,j2)*CChat(i2,j2,k2,l2)*PP1(k,l,k2,l2)
                    end do
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
      CC_ref = CCvol + CCiso

      do i=1,3
        do j=1,3
          do k=1,3
            do l=1,3
              cc_def(i,j,k,l) = 0.
              do i2=1,3
                do j2=1,3
                  do k2=1,3
                    do l2=1,3
                      cc_def(i,j,k,l) = cc_def(i,j,k,l) +
     #                              dfgrd1(i,i2)*dfgrd1(j,j2)*dfgrd1(k,k2)*
     #                              dfgrd1(l,l2)*CC_ref(i2,j2,k2,l2)
                    end do
                  end do
                end do
              end do
              cc_def(i,j,k,l) = cc_def(i,j,k,l)/detf
            end do
          end do
        end do
      end do

c...  Abaqus corrections 
      do II=1,6
        do JJ=1,6
          i = Itoi(II)
          j = Itoj(II)
          k = Itoi(JJ)
          l = Itoj(JJ)
          ddsdde(II,JJ) = cc_def(i,j,k,l)+0.5*(kron(i,k)*sigma(j,l)
     #                    +kron(i,l)*sigma(j,k)+kron(j,k)*sigma(i,l)
     #                    +kron(j,l)*sigma(i,k))
          if (JJ>II) then
            ddsdde(JJ,II) = ddsdde(II,JJ)
          end if
        end do
      end do

c      if (noel.eq.500.and.npt.eq.1) then
c        write(7,*) "CChat"
c        write(7,*) CChat
c        write(7,*) "dShat/dC"
c        write(7,*) dShatdC
c        write(7,*) "CCvol"
c        write(7,*) CCvol
c        write(7,*) "CCiso"
c        write(7,*) CCiso
c        write(7,*) "CC_ref"
c        write(7,*) CC_ref
c        write(7,*) "DDSDDE"
c        write(7,*) ddsdde
c      end if
c...  calculate strain energy
      sse = Psivol


      return
      contains

      real*8 function det(C)
        implicit none
        real*8 C(3,3)

        det = +C(1,1)*(C(2,2)*C(3,3)-C(2,3)*C(3,2))
     #        -C(1,2)*(C(2,1)*C(3,3)-C(2,3)*C(3,1))
     #        +C(1,3)*(C(2,1)*C(3,2)-C(2,2)*C(3,1))
      end

      end

c...  ------------------------------------------------------------------

      subroutine activation(value, typea)
      implicit none

      real*8 value
      integer typea
c...  typea = 0; ReLU, = 1; sigmoid, = 2; linear
      if (typea==0) then
        if (value<0) then
          value = 0
        end if
      else if (typea==1) then
        value = 1/(1+exp(-value))
      end if

      return
      end
c...  ------------------------------------------------------------------

      subroutine grad_activation(value, typea)
      implicit none

      real*8 value
      real*8 sigmoid
      integer typea

      if (typea==0) then
        if (value<1e-9) then
          value = 0
        else
          value = 1
        end if
      else if (typea==1) then
        sigmoid =1/(1+exp(value))
        value = sigmoid*(1-sigmoid)
      else
        value = 1
      end if

      return
      end

c...  ------------------------------------------------------------------


c...  ------------------------------------------------------------------
      end
c...  ------------------------------------------------------------------
