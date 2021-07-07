/* 
	Constitutive Equations

	For the mechanics
*/
//
//------------------------------------------------------------------------------------//
// Last edit on 02 August 2018 by Adrian Buganza
// EDIT: removed phic/thetaP, this was added for the conversion of col volume fraction
//
//------------------------------------------------------------------------------------//
// Last edit on 03 July 2018 by Marco Pensalfini
// EDIT: removed fibronectin terms from both functions, but still passed to keep structure
// EDIT: removed hard-coded parameters in evalSS_pas; they are now passed from global_parameters
// EDIT: global_parameters now passed also to evalDD_pas (b/c evalSS_pas is called there)
//------------------------------------------------------------------------------------//
//
#include "mechanics.h"
#include <vector>
#include <map>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element
using namespace Eigen;

//------------------------------------------------------------------------------------//
// 	biology of global variables
//------------------------------------------------------------------------------------//

// for the flux function
// This defines the passive (2nd PK) stress term
Matrix2d evalSS_pas(
	const Matrix2d &FF, double rho, const Vector2d &Grad_rho, double c, const Vector2d &Grad_c,
	double phic, double kc, const Vector2d &a0c, double kappac,
	double phif, double kf, const Vector2d &a0f, double kappaf,const Vector2d &lamdaP, const std::vector<double> &parameters)
{

	// parameters
	// double mu0bar = parameters[0];
	// double kcbar = parameters[27];
	// //double kcbar = 5.581;
	// double alphaf = 150.; 
	// double mu0 = (alphaf*phif+1.)*mu0bar;
	// //double mu0 = mu0bar;
	// if(phic<0.001 && phif<0.001){mu0=1e-09;}
	// // rewrite the value of kc here based on phic 
	// kc = phic*phic*kcbar; 
	// double k2c = parameters[1];
	// //double k2f = parameters[2];
	// //
	// // make k0 dependent on phic up to a threshold, useful for wounding
	// double mu0min = 1e-09;
	// double phic_thresh = 0.1;
	//if(phic < phic_thresh){mu0=(mu0-mu0min)/phic_thresh*phic+mu0min;}
	//
	// re-compute basis a0, s0
	Matrix2d CC = FF.transpose()*FF;
	Matrix2d CCinv = CC.inverse();
	Matrix2d Identity = Matrix2d::Identity(2,2);
	// double lamdaP_a = lamdaP(0);
	// double lamdaP_s = lamdaP(1);
	// //
	// Matrix2d Rot90;Rot90<<0.,-1.,1.,0.;
	// Vector2d s0c = Rot90*a0c;
	// //Vector2d s0f = Rot90*a0f;
	// // fiber tensor in the reference
	// Matrix2d a0ca0c = a0c*a0c.transpose();
	// Matrix2d s0cs0c = s0c*s0c.transpose();
	// //
	// // recompute split. The contraction occurs with respect to the collagen direction
	// Matrix2d FFg = lamdaP_a*(a0ca0c) + lamdaP_s*(s0cs0c);
	// double thetaP = lamdaP_a*lamdaP_s;
	// Matrix2d FFginv = (1./lamdaP_a)*(a0ca0c) + (1./lamdaP_s)*(s0cs0c);
	// Matrix2d FFe = FF*FFginv;
	// //std::cout<<"recompute the split.\nFF\n"<<FF<<"\nFg\n"<<FFg<<"\nFFe\n"<<FFe<<"\n";
	// // elastic strain
	// Matrix2d CCe = FFe.transpose()*FFe;
	// double I1e2D = CCe(0,0) + CCe(1,1);
	// double I4ce2D = a0c.dot(CCe*a0c);
	// //
	// // now consider full 3D case
	// Matrix3d Identity3D = Matrix3d::Identity(3,3);
	// // fiber tensors
	// Vector3d a0c3D; a0c3D<<a0c(0),a0c(1),0.0;
	// Vector3d s0c3D; s0c3D<<s0c(0),s0c(1),0.0;
	// Matrix3d a0ca0c3D = a0c3D*a0c3D.transpose();
	// Matrix3d s0cs0c3D = s0c3D*s0c3D.transpose();
	// // CC and invariants
	// double CCe33 = 1./CCe.determinant();
	// Matrix3d CCe3D;CCe3D<<CCe(0,0),CCe(0,1),0.,CCe(1,0),CCe(1,1),0.,0.,0.,CCe33;
	// Matrix3d CCe3Dinv = CCe3D.inverse();
	// double CCe33inv = CCe3Dinv(2,2);
	// double I1e3D = CCe3D(0,0) + CCe3D(1,1) + CCe3D(2,2);
	// double I4ce3D = I4ce2D;
	// //
	// // PK2 stress tensor under plane stress assumption (Sigma_33 = 0)
	// // matrix contribution
	// Matrix3d SSe_m = mu0*(Identity3D-1./3.*I1e3D*CCe3Dinv);
	// // fiber contribution
	// double Ebar = kappac*I1e3D+(1.-3.*kappac)*I4ce3D-1.;
	// double Psic = kc/(2.*k2c)*exp( k2c*pow(Ebar,2) );
	// Matrix3d SSe_f = 4.*k2c*Ebar*Psic*(kappac*Identity3D + (1.-3.*kappac)*a0ca0c3D - 1./3.*(kappac*I1e3D+(1.-3.*kappac)*I4ce3D)*CCe3Dinv );
	// // // volumetric term
	// Matrix3d SSe_v = (1./3.*I1e3D-1./CCe33inv)*(mu0+2.*kappac*kc*(kappac*I1e3D-1.)*exp(k2c*pow(kappac*I1e3D-1.,2)))*CCe3Dinv;
	// //
	// // get 3D PK2 stress matrix
	// Matrix3d SSe3D_pas = SSe_m + SSe_f + SSe_v;
	// // slice down to 2D matrix
	// // Matrix2d SSe_pas; SSe_pas<<SSe3D_pas(0,0),SSe3D_pas(0,1),SSe3D_pas(1,0),SSe3D_pas(1,1);
	// Matrix2d SSe_pas; SSe_pas.setZero();
	// // pull back the whole 2D thing to the reference
	// // Matrix2d SS = thetaP*FFginv*SSe_pas*FFginv;
	Matrix2d SS;

	// St venant kirchhoff to check convergence of a linear material
	// St Venant Kirchhoff is linear for a description in the reference
	// i.e. in terms of S and E rather than linear in sigma and epsilon
	// Not good for large deformations, but good to check convergence
	
	double nu = 0.4;
	double mu0 = 0.1;
	double Ym = mu0;
	Matrix3d SVK; SVK.setZero(); SVK<<1,nu,0,nu,1,0,0,0,(1-nu)/2.;
	SVK = mu0/(1.0-nu*nu)*SVK;
	Matrix2d EE = 0.5*(CC-Identity);
	Vector3d EEvoigt; EEvoigt.setZero(); EEvoigt<<EE(0,0),EE(1,1),2*EE(0,1);
	Vector3d SSvoigt = SVK*EEvoigt;
	SS(0,0) = SSvoigt(0);
	SS(1,1) = SSvoigt(1);
	SS(0,1) = SSvoigt(2);
	SS(1,0) = SSvoigt(2);
	// std::cout<<"CC matrix\n"<<CC<<"\n";
	// std::cout<<"EE matrix\n"<<EEvoigt<<"\n";
	// std::cout<<"SVK\n"<<SVK<<"\n";
	
	return SS;
}

// This computes the numerical tangent of the passive stress term
Matrix3d eval_DDpas(
	const Matrix2d &FF, double rho, const Vector2d &Grad_rho, double c, const Vector2d &Grad_c,
	double phic, double kc, const Vector2d &a0c, double kappac,
	double phif, double kf, const Vector2d &a0f, double kappaf,const Vector2d &lamdaP, const std::vector<double> &parameters)
{
	Matrix3d DDpas;DDpas.setZero();
	// Mechanics
	// numerical
	Matrix2d CC = FF.transpose()*FF;
	EigenSolver<Matrix2d> es(CC);
	// then polar decomposition is
	std::vector<Vector2d> Ubasis(2,Vector2d(0.0,0.0));
	Matrix2d CC_eiv = es.eigenvectors().real();
	Ubasis[0] =  CC_eiv.col(0);
	Ubasis[1] =  CC_eiv.col(1);
	Vector2d Clamda =  es.eigenvalues().real();
	// Build the U matrix from basis and the root of the eigenvuales
	Matrix2d UU = sqrt(Clamda(0))*Ubasis[0]*Ubasis[0].transpose() + sqrt(Clamda(1))*Ubasis[1]*Ubasis[1].transpose() ;
	Matrix2d RR = FF*UU.inverse();
	Matrix2d FF_plus,FF_minus;
	Matrix2d CC_plus,CC_minus;
	Matrix2d UU_plus,UU_minus;
	std::vector<Vector2d> Ebasis(2,Vector2d(0.0,0.0));
	Ebasis[0] = Vector2d(1.0,0.0);
	Ebasis[1] = Vector2d(0.0,1.0);
	std::vector<Vector2d> Ubasis_plus(2,Vector2d(0.0,0.0));
	std::vector<Vector2d> Ubasis_minus(2,Vector2d(0.0,0.0));
	Matrix2d CC_eiv_plus,CC_eiv_minus;
	Vector2d Clamda_plus,Clamda_minus;
	//
	Matrix2d SS_plus,SS_minus;
	//

	double epsilon = 1e-9;

	//
	Vector3d voigt_table_I_i(0,1,0);
	Vector3d voigt_table_I_j(0,1,1);
	Vector3d voigt_table_J_k(0,1,0);
	Vector3d voigt_table_J_l(0,1,1);
	int ii,jj,kk,ll;
	for(int II=0;II<3;II++){
		for(int JJ=0;JJ<3;JJ++){
			ii = voigt_table_I_i(II);
			jj = voigt_table_I_j(II);
			kk = voigt_table_J_k(JJ);
			ll = voigt_table_J_l(JJ);

			CC_plus = CC + epsilon*(Ebasis[kk]*Ebasis[ll].transpose())+ epsilon*(Ebasis[ll]*Ebasis[kk].transpose());
			CC_minus = CC - epsilon*(Ebasis[kk]*Ebasis[ll].transpose())- epsilon*(Ebasis[ll]*Ebasis[kk].transpose());

			// Polar decomposition of CCplus and CCminus
			EigenSolver<Matrix2d> esp(CC_plus);
			EigenSolver<Matrix2d> esm(CC_minus);
			Matrix2d CC_eiv_plus = esp.eigenvectors().real();
			Matrix2d CC_eiv_minus = esm.eigenvectors().real();
			Ubasis_plus[0] =  CC_eiv_plus.col(0);
			Ubasis_plus[1] =  CC_eiv_plus.col(1);
			Ubasis_minus[0] =  CC_eiv_minus.col(0);
			Ubasis_minus[1] =  CC_eiv_minus.col(1);
			Clamda_plus =  esp.eigenvalues().real();
			Clamda_minus =  esm.eigenvalues().real();
			UU_plus = sqrt(Clamda_plus(0))*Ubasis_plus[0]*Ubasis_plus[0].transpose() + sqrt(Clamda_plus(1))*Ubasis_plus[1]*Ubasis_plus[1].transpose() ;
			UU_minus = sqrt(Clamda_minus(0))*Ubasis_minus[0]*Ubasis_minus[0].transpose() + sqrt(Clamda_minus(1))*Ubasis_minus[1]*Ubasis_minus[1].transpose() ;
			FF_plus = RR*UU_plus;
			FF_minus = RR*UU_minus;

			SS_plus =  evalSS_pas(FF_plus,rho,Grad_rho,c,Grad_c,phic,kc,a0c,kappac,phif,kf,a0f,kappaf,lamdaP,parameters);
			SS_minus =  evalSS_pas(FF_minus,rho,Grad_rho,c,Grad_c,phic,kc,a0c,kappac,phif,kf,a0f,kappaf,lamdaP,parameters);

			DDpas(II,JJ) =  (1.0/(4.0*epsilon))*(SS_plus(ii,jj) - SS_minus(ii,jj) );
		}
	}
	return DDpas;
}
