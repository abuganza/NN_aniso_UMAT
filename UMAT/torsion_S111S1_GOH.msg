1

   Abaqus 2020                                  Date 16-Jun-2021   Time 16:12:00
   For use by PURDUE UNIVERSITY under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                0.100    
          AND A TOTAL TIME PERIOD OF                            10.0    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-04
          THE MAXIMUM TIME INCREMENT ALLOWED IS                 10.0    

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE MOMENT                              1.000E-02
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                     100

          LARGE DISPLACEMENT THEORY WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH      3.07    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  7.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  2.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  9.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.896E-03   TIME AVG. FORCE       1.896E-03
 LARGEST RESIDUAL FORCE            -0.477       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.250       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.250       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.072E-02
                                                (FROM FORCE     FIELD)
 LARGEST RESIDUAL MOMENT           -4.229E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  2.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.786E-06   TIME AVG. FORCE       6.786E-06
 LARGEST RESIDUAL FORCE            -1.803E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.250       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.263E-03   AT NODE        108   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.072E-02
                                                (FROM FORCE     FIELD)
 LARGEST RESIDUAL MOMENT           -3.020E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.779E-14   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.785E-06   TIME AVG. FORCE       6.785E-06
 LARGEST RESIDUAL FORCE             2.253E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.250       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.927E-04   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.072E-02
                                                (FROM FORCE     FIELD)
 LARGEST RESIDUAL MOMENT           -2.306E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.940E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.785E-06   TIME AVG. FORCE       6.785E-06
 LARGEST RESIDUAL FORCE             6.626E-12   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.250       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.475E-09   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.072E-02
                                                (FROM FORCE     FIELD)
 LARGEST RESIDUAL MOMENT            7.371E-12   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.181E-16   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  1.000E-02
 STEP TIME COMPLETED       0.100    ,  TOTAL TIME COMPLETED        0.100    


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.892E-04   TIME AVG. FORCE       1.480E-04
 LARGEST RESIDUAL FORCE              1.00       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.251       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.505E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.876E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.698E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.357E-05   TIME AVG. FORCE       1.018E-05
 LARGEST RESIDUAL FORCE             4.777E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.251       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.479E-03   AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.127E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.263E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.357E-05   TIME AVG. FORCE       1.018E-05
 LARGEST RESIDUAL FORCE             2.530E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.251       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.439E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.833E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.732E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.357E-05   TIME AVG. FORCE       1.018E-05
 LARGEST RESIDUAL FORCE            -1.570E-11   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.251       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.521E-08   AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.632E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.000E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.321E-15   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.150    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  2.000E-02
 STEP TIME COMPLETED       0.200    ,  TOTAL TIME COMPLETED        0.200    


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.150    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 52 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.752E-04   TIME AVG. FORCE       1.985E-04
 LARGEST RESIDUAL FORCE              1.89       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.377       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.317E-02   AT NODE        397   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.951E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.500E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.824E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.391E-05   TIME AVG. FORCE       2.476E-05
 LARGEST RESIDUAL FORCE             7.656E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.377       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.174E-02   AT NODE        451   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.123E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.500E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.055E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.382E-05   TIME AVG. FORCE       1.473E-05
 LARGEST RESIDUAL FORCE             8.245E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.377       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.558E-03   AT NODE        127   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.865E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.500E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.723E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.376E-05   TIME AVG. FORCE       1.471E-05
 LARGEST RESIDUAL FORCE             3.974E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.377       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.058E-05   AT NODE        162   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.198E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.500E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.676E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.225    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.150    ,  FRACTION OF STEP COMPLETED  3.500E-02
 STEP TIME COMPLETED       0.350    ,  TOTAL TIME COMPLETED        0.350    


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.225    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 249 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.991E-03   TIME AVG. FORCE       5.087E-04
 LARGEST RESIDUAL FORCE              4.22       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        9.294E-02   AT NODE        722   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.158E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.891E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 16 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.958E-04   TIME AVG. FORCE       1.600E-04
 LARGEST RESIDUAL FORCE            -4.171E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.971E-02   AT NODE        722   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.313E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.195E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.003E-04   TIME AVG. FORCE       3.611E-05
 LARGEST RESIDUAL FORCE            -8.053E-03   AT NODE        827   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.710E-02   AT NODE        794   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.157E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.969E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.190E-05   TIME AVG. FORCE       2.150E-05
 LARGEST RESIDUAL FORCE             7.855E-04   AT NODE        702   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.403E-02   AT NODE        810   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.461E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.394E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.911E-05   TIME AVG. FORCE       2.081E-05
 LARGEST RESIDUAL FORCE             7.704E-05   AT NODE        668   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.752E-03   AT NODE        721   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.626E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.393E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      3.905E-05   TIME AVG. FORCE       2.079E-05
 LARGEST RESIDUAL FORCE            -1.694E-07   AT NODE        667   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.990E-04   AT NODE        721   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.384E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.499E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      3.905E-05   TIME AVG. FORCE       2.079E-05
 LARGEST RESIDUAL FORCE             8.747E-12   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.153E-06   AT NODE        721   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.071E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.388E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   7 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  7 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.225    ,  FRACTION OF STEP COMPLETED  5.750E-02
 STEP TIME COMPLETED       0.575    ,  TOTAL TIME COMPLETED        0.575    


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.225    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 291 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.152E-03   TIME AVG. FORCE       4.469E-04
 LARGEST RESIDUAL FORCE              5.06       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.569       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -9.284E-02   AT NODE        919   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.692E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.091E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 24 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.433E-03   TIME AVG. FORCE       3.033E-04
 LARGEST RESIDUAL FORCE             0.105       AT NODE        758   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.569       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.230       AT NODE        864   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.787E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.200E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 53 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.182E-02   TIME AVG. FORCE       2.381E-03
 LARGEST RESIDUAL FORCE              2.79       AT NODE        631   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.833       AT NODE        631   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.782       AT NODE        631   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.648E-03   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.047E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 151 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.250E-02   TIME AVG. FORCE       6.516E-03
 LARGEST RESIDUAL FORCE             -4.00       AT NODE        989   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.934       AT NODE        612   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.902       AT NODE       1080   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT             1.02       AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.250E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.682E-03   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     5 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  5.625E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.576E-04   TIME AVG. FORCE       6.816E-05
 LARGEST RESIDUAL FORCE             0.791       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.142       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.995E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.042E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.625E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.468E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.288E-05   TIME AVG. FORCE       2.521E-05
 LARGEST RESIDUAL FORCE             4.184E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.142       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.170E-03   AT NODE         92   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.729E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.625E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.093E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.288E-05   TIME AVG. FORCE       2.521E-05
 LARGEST RESIDUAL FORCE             1.800E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.142       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        9.397E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.720E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.625E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.503E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.288E-05   TIME AVG. FORCE       2.521E-05
 LARGEST RESIDUAL FORCE             1.654E-11   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.142       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.919E-09   AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.583E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.625E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.380E-15   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.625E-02,  FRACTION OF STEP COMPLETED  6.313E-02
 STEP TIME COMPLETED       0.631    ,  TOTAL TIME COMPLETED        0.631    


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.625E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.298E-04   TIME AVG. FORCE       4.264E-05
 LARGEST RESIDUAL FORCE             0.316       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.142       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.994E-04   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            9.578E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.625E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.328E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.671E-05   TIME AVG. FORCE       2.879E-05
 LARGEST RESIDUAL FORCE             9.987E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.142       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.842E-04   AT NODE         92   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            7.202E-12   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.625E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.808E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.671E-05   TIME AVG. FORCE       2.879E-05
 LARGEST RESIDUAL FORCE             9.670E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.142       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.605E-05   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.216E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.625E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.608E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    8.438E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.625E-02,  FRACTION OF STEP COMPLETED  6.875E-02
 STEP TIME COMPLETED       0.688    ,  TOTAL TIME COMPLETED        0.688    


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.438E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.115E-04   TIME AVG. FORCE       5.490E-05
 LARGEST RESIDUAL FORCE             0.593       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.213       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.500E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.805E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.438E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.664E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.246E-05   TIME AVG. FORCE       3.217E-05
 LARGEST RESIDUAL FORCE             2.866E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.213       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.899E-04   AT NODE         92   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.595E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.438E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.931E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.246E-05   TIME AVG. FORCE       3.217E-05
 LARGEST RESIDUAL FORCE             8.221E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.213       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.426E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.997E-12   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.438E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.899E-14   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.127    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.438E-02,  FRACTION OF STEP COMPLETED  7.719E-02
 STEP TIME COMPLETED       0.772    ,  TOTAL TIME COMPLETED        0.772    


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.127    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.240E-04   TIME AVG. FORCE       8.116E-05
 LARGEST RESIDUAL FORCE              1.33       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.321       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.538E-03   AT NODE        234   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.560E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.266E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.517E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.112E-05   TIME AVG. FORCE       3.579E-05
 LARGEST RESIDUAL FORCE            -4.575E-05   AT NODE        270   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.321       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.718E-03   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.731E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.266E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.115E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.111E-05   TIME AVG. FORCE       3.579E-05
 LARGEST RESIDUAL FORCE             1.381E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.321       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.249E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.374E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.266E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.562E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.110E-05   TIME AVG. FORCE       3.579E-05
 LARGEST RESIDUAL FORCE             1.642E-11   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.321       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.253E-07   AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.472E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.266E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.212E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.190    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.127    ,  FRACTION OF STEP COMPLETED  8.984E-02
 STEP TIME COMPLETED       0.898    ,  TOTAL TIME COMPLETED        0.898    


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.190    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 170 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.962E-04   TIME AVG. FORCE       1.314E-04
 LARGEST RESIDUAL FORCE              3.00       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.482       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.128E-02   AT NODE         81   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.996E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.898E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.281E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.521E-05   TIME AVG. FORCE       4.017E-05
 LARGEST RESIDUAL FORCE             2.757E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.482       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.081E-02   AT NODE         81   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.039E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.898E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.763E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.410E-05   TIME AVG. FORCE       4.005E-05
 LARGEST RESIDUAL FORCE             1.791E-05   AT NODE        179   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.482       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.600E-03   AT NODE        127   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.646E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.898E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.507E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.411E-05   TIME AVG. FORCE       4.005E-05
 LARGEST RESIDUAL FORCE             1.446E-10   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.482       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        9.963E-06   AT NODE        559   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.127E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.898E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.334E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.285    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.190    ,  FRACTION OF STEP COMPLETED  0.109    
 STEP TIME COMPLETED        1.09    ,  TOTAL TIME COMPLETED         1.09    


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.285    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 348 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.997E-03   TIME AVG. FORCE       2.358E-04
 LARGEST RESIDUAL FORCE              6.75       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.724       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.724E-02   AT NODE       1008   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.239E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.848E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.875E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 9 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.839E-04   TIME AVG. FORCE       5.443E-05
 LARGEST RESIDUAL FORCE            -4.447E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.724       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.677E-02   AT NODE         90   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.920E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.848E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.280E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.006E-04   TIME AVG. FORCE       4.610E-05
 LARGEST RESIDUAL FORCE            -1.777E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.724       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.562E-02   AT NODE         90   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.269E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.848E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.787E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.372E-05   TIME AVG. FORCE       4.542E-05
 LARGEST RESIDUAL FORCE            -8.183E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.724       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.343E-03   AT NODE        127   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.090E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.848E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.938E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      9.372E-05   TIME AVG. FORCE       4.542E-05
 LARGEST RESIDUAL FORCE            -1.618E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.724       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        8.791E-05   AT NODE        127   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.347E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.848E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.358E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.285    ,  FRACTION OF STEP COMPLETED  0.137    
 STEP TIME COMPLETED        1.37    ,  TOTAL TIME COMPLETED         1.37    


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.285    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 382 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.433E-03   TIME AVG. FORCE       4.442E-04
 LARGEST RESIDUAL FORCE              8.09       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.725       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.373       AT NODE       1008   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.280E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.848E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.001E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 54 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT PART-1-1.563 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.611 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.614 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.662 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.732 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.748 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 8 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    11 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  7.119E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.379E-04   TIME AVG. FORCE       8.109E-05
 LARGEST RESIDUAL FORCE              1.27       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.181       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.223E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            9.412E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.119E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.898E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.865E-05   TIME AVG. FORCE       5.026E-05
 LARGEST RESIDUAL FORCE             2.146E-05   AT NODE        108   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.181       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.472E-03   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.642E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.119E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.676E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      9.864E-05   TIME AVG. FORCE       5.025E-05
 LARGEST RESIDUAL FORCE             4.546E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.181       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.101E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.018E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.119E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.581E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.864E-05   TIME AVG. FORCE       5.025E-05
 LARGEST RESIDUAL FORCE            -2.518E-11   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.181       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.075E-07   AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.587E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.119E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.378E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.119E-02,  FRACTION OF STEP COMPLETED  0.144    
 STEP TIME COMPLETED        1.44    ,  TOTAL TIME COMPLETED         1.44    


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.119E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.332E-04   TIME AVG. FORCE       6.550E-05
 LARGEST RESIDUAL FORCE             0.507       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.181       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.290E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.402E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.119E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.249E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.036E-04   TIME AVG. FORCE       5.470E-05
 LARGEST RESIDUAL FORCE             2.244E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.181       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.589E-04   AT NODE        125   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.066E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.119E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.227E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.036E-04   TIME AVG. FORCE       5.470E-05
 LARGEST RESIDUAL FORCE             4.973E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.181       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.011E-05   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            9.478E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.119E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.500E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.107    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.119E-02,  FRACTION OF STEP COMPLETED  0.152    
 STEP TIME COMPLETED        1.52    ,  TOTAL TIME COMPLETED         1.52    


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.107    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.616E-04   TIME AVG. FORCE       7.831E-05
 LARGEST RESIDUAL FORCE             0.950       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.272       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.420E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.924E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.068E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.588E-13   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.110E-04   TIME AVG. FORCE       5.903E-05
 LARGEST RESIDUAL FORCE             4.816E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.272       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.375E-03   AT NODE        125   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.898E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.068E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.434E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.110E-04   TIME AVG. FORCE       5.903E-05
 LARGEST RESIDUAL FORCE             2.478E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.272       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.315E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.312E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.068E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.524E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.160    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.107    ,  FRACTION OF STEP COMPLETED  0.162    
 STEP TIME COMPLETED        1.62    ,  TOTAL TIME COMPLETED         1.62    


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.160    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 84 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.009E-04   TIME AVG. FORCE       1.049E-04
 LARGEST RESIDUAL FORCE              2.14       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.408       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.519E-02   AT NODE         73   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.344E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.602E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.404E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.261E-04   TIME AVG. FORCE       6.382E-05
 LARGEST RESIDUAL FORCE             2.281E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.408       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.343E-02   AT NODE         73   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.779E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.602E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.512E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.222E-04   TIME AVG. FORCE       6.354E-05
 LARGEST RESIDUAL FORCE             1.560E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.408       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.863E-03   AT NODE        127   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.666E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.602E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.754E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.222E-04   TIME AVG. FORCE       6.354E-05
 LARGEST RESIDUAL FORCE            -6.082E-10   AT NODE        451   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.408       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.516E-05   AT NODE        576   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.544E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.602E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.660E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.240    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.160    ,  FRACTION OF STEP COMPLETED  0.178    
 STEP TIME COMPLETED        1.78    ,  TOTAL TIME COMPLETED         1.78    


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.240    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 275 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.593E-03   TIME AVG. FORCE       1.655E-04
 LARGEST RESIDUAL FORCE              4.81       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.613       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.662E-02   AT NODE        810   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.462E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.403E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.719E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.810E-03   TIME AVG. FORCE       1.800E-04
 LARGEST RESIDUAL FORCE             -1.98       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.682       AT NODE         73   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.417       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.658E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.403E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.730E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  6.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 37 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.016E-04   TIME AVG. FORCE       9.941E-05
 LARGEST RESIDUAL FORCE             0.232       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.699       AT NODE        126   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.119       AT NODE        144   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.118E-02   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.403E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.467E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      9.717E-03   TIME AVG. FORCE       7.071E-04
 LARGEST RESIDUAL FORCE             -7.00       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         -1.40       AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        -1.01       AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT             2.73       AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.403E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.360E-04   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    15 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  6.007E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.958E-04   TIME AVG. FORCE       7.902E-05
 LARGEST RESIDUAL FORCE             0.661       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.153       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.686E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.287E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.007E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.507E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.264E-04   TIME AVG. FORCE       6.773E-05
 LARGEST RESIDUAL FORCE             3.363E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.153       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        9.763E-04   AT NODE        125   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.582E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.007E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.454E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.264E-04   TIME AVG. FORCE       6.773E-05
 LARGEST RESIDUAL FORCE             1.135E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.153       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.696E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.735E-12   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.007E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.688E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    9.010E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.007E-02,  FRACTION OF STEP COMPLETED  0.184    
 STEP TIME COMPLETED        1.84    ,  TOTAL TIME COMPLETED         1.84    


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.010E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.055E-04   TIME AVG. FORCE       8.259E-05
 LARGEST RESIDUAL FORCE             0.676       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.230       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.725E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.056E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.010E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.021E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.327E-04   TIME AVG. FORCE       7.179E-05
 LARGEST RESIDUAL FORCE             3.470E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.230       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -9.954E-04   AT NODE         92   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.130E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.010E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.282E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.327E-04   TIME AVG. FORCE       7.179E-05
 LARGEST RESIDUAL FORCE             1.210E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.230       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.979E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.005E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.010E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.029E-13   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.135    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.010E-02,  FRACTION OF STEP COMPLETED  0.193    
 STEP TIME COMPLETED        1.93    ,  TOTAL TIME COMPLETED         1.93    


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.135    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 13 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.128E-04   TIME AVG. FORCE       1.095E-04
 LARGEST RESIDUAL FORCE              1.55       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.345       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        8.201E-02   AT NODE        216   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.233E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.711E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.840E-04   TIME AVG. FORCE       8.428E-05
 LARGEST RESIDUAL FORCE            -2.384E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.394       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.119       AT NODE        199   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.538E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.710E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.634E-04   TIME AVG. FORCE       7.718E-05
 LARGEST RESIDUAL FORCE            -1.035E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.373       AT NODE        199   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.652E-02   AT NODE        217   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.752E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.666E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.553E-04   TIME AVG. FORCE       7.671E-05
 LARGEST RESIDUAL FORCE            -4.052E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.345       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.036E-02   AT NODE        199   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.760E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.668E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.423E-04   TIME AVG. FORCE       7.594E-05
 LARGEST RESIDUAL FORCE            -1.672E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.345       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.456E-03   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.246E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.462E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  2.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      1.422E-04   TIME AVG. FORCE       7.594E-05
 LARGEST RESIDUAL FORCE             3.753E-06   AT NODE        162   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.345       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.079E-03   AT NODE        162   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.287E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.675E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      1.422E-04   TIME AVG. FORCE       7.594E-05
 LARGEST RESIDUAL FORCE             2.089E-10   AT NODE        342   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.345       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.046E-06   AT NODE        162   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.000E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.420E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   7 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  7 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.135    ,  FRACTION OF STEP COMPLETED  0.207    
 STEP TIME COMPLETED        2.07    ,  TOTAL TIME COMPLETED         2.07    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.135    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 43 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.532E-04   TIME AVG. FORCE       1.080E-04
 LARGEST RESIDUAL FORCE              1.83       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.346       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.868E-02   AT NODE        127   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.146E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.080E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.659E-04   TIME AVG. FORCE       8.094E-05
 LARGEST RESIDUAL FORCE             4.296E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.345       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.508E-02   AT NODE        487   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.043E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.507E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.519E-04   TIME AVG. FORCE       8.015E-05
 LARGEST RESIDUAL FORCE             4.677E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.345       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.349E-03   AT NODE         90   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.654E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.386E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.518E-04   TIME AVG. FORCE       8.015E-05
 LARGEST RESIDUAL FORCE            -3.232E-08   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.345       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.963E-04   AT NODE        487   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.683E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.443E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.135    ,  FRACTION OF STEP COMPLETED  0.220    
 STEP TIME COMPLETED        2.20    ,  TOTAL TIME COMPLETED         2.20    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.135    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 42 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.274E-04   TIME AVG. FORCE       1.142E-04
 LARGEST RESIDUAL FORCE              1.85       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.354       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.453E-02   AT NODE         90   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.937E-03   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.439E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.111E-04   TIME AVG. FORCE       8.704E-05
 LARGEST RESIDUAL FORCE             9.000E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.346       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.872E-02   AT NODE        487   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.098E-03   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.093E-05   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.929E-04   TIME AVG. FORCE       8.608E-05
 LARGEST RESIDUAL FORCE            -7.792E-03   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.345       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.843E-02   AT NODE        415   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.997E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.421E-05   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  6.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  2.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  8.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.625E-04   TIME AVG. FORCE       8.448E-05
 LARGEST RESIDUAL FORCE             8.215E-04   AT NODE        378   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.345       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.940E-02   AT NODE        450   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.765E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.228E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      1.615E-04   TIME AVG. FORCE       8.443E-05
 LARGEST RESIDUAL FORCE            -3.282E-04   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.344       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.008E-02   AT NODE        415   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.438E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.065E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      1.614E-04   TIME AVG. FORCE       8.443E-05
 LARGEST RESIDUAL FORCE             1.593E-06   AT NODE        378   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.344       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.367E-04   AT NODE        378   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.654E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.598E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      1.614E-04   TIME AVG. FORCE       8.443E-05
 LARGEST RESIDUAL FORCE             1.075E-09   AT NODE        378   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.344       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.840E-05   AT NODE        378   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.724E-10   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.592E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   7 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  7 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.135    ,  FRACTION OF STEP COMPLETED  0.234    
 STEP TIME COMPLETED        2.34    ,  TOTAL TIME COMPLETED         2.34    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.135    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 41 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.995E-03   TIME AVG. FORCE       1.800E-04
 LARGEST RESIDUAL FORCE              2.11       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.393       AT NODE        415   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.183       AT NODE        487   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.496E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.152E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 38 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.021E-02   TIME AVG. FORCE       5.908E-04
 LARGEST RESIDUAL FORCE             -4.88       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.662       AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.874       AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT             1.25       AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.721E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 125 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.328E-02   TIME AVG. FORCE       2.244E-03
 LARGEST RESIDUAL FORCE              8.88       AT NODE        538   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         -1.13       AT NODE        162   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.966       AT NODE         90   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT             4.41       AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.352E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.598E-03   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 279 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.31 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.38 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.52 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.52 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.52 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.64 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.64 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.72 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.82 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.98 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.99 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.99 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.115 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.136 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.136 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.149 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.150 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.150 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.150 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.184 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.186 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.192 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.200 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.200 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.235 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.237 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.241 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.251 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.268 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.269 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.285 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.292 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.308 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.309 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.326 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.343 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.354 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.371 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.388 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.389 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.394 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.411 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.412 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.426 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.426 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.426 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.426 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.509 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.509 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.546 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.546 IS DISTORTING EXCESSIVELY.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 111 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    20 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  3.379E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.262E-04   TIME AVG. FORCE       9.152E-05
 LARGEST RESIDUAL FORCE             0.285       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.608E-02   AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.273E-04   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.451E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.379E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.821E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.638E-04   TIME AVG. FORCE       8.840E-05
 LARGEST RESIDUAL FORCE             8.288E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.608E-02   AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.279E-04   AT NODE         74   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.093E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.379E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.450E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.638E-04   TIME AVG. FORCE       8.840E-05
 LARGEST RESIDUAL FORCE             6.597E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -8.608E-02   AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.317E-05   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.526E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.379E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.518E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.379E-02,  FRACTION OF STEP COMPLETED  0.237    
 STEP TIME COMPLETED        2.37    ,  TOTAL TIME COMPLETED         2.37    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.379E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.837E-04   TIME AVG. FORCE       9.294E-05
 LARGEST RESIDUAL FORCE             0.114       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -8.607E-02   AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.908E-04   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.298E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.379E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.926E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.663E-04   TIME AVG. FORCE       9.211E-05
 LARGEST RESIDUAL FORCE             1.472E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.607E-02   AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.746E-04   AT NODE         74   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.607E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.379E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.408E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.663E-04   TIME AVG. FORCE       9.211E-05
 LARGEST RESIDUAL FORCE             2.166E-10   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.607E-02   AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.168E-06   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.936E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.379E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.596E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    5.068E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.379E-02,  FRACTION OF STEP COMPLETED  0.241    
 STEP TIME COMPLETED        2.41    ,  TOTAL TIME COMPLETED         2.41    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.068E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.123E-04   TIME AVG. FORCE       9.757E-05
 LARGEST RESIDUAL FORCE             0.214       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.129       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.452E-04   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.307E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.068E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.653E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.699E-04   TIME AVG. FORCE       9.564E-05
 LARGEST RESIDUAL FORCE             4.873E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.129       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.228E-04   AT NODE         74   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.440E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.068E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.692E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.699E-04   TIME AVG. FORCE       9.564E-05
 LARGEST RESIDUAL FORCE             2.273E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.129       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.473E-06   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.133E-12   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.068E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.397E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    7.602E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.068E-02,  FRACTION OF STEP COMPLETED  0.246    
 STEP TIME COMPLETED        2.46    ,  TOTAL TIME COMPLETED         2.46    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.602E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.897E-04   TIME AVG. FORCE       1.041E-04
 LARGEST RESIDUAL FORCE             0.482       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.194       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.226E-03   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.564E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.602E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.788E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.753E-04   TIME AVG. FORCE       9.911E-05
 LARGEST RESIDUAL FORCE             2.068E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.194       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.073E-04   AT NODE         92   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.524E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.602E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.180E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.754E-04   TIME AVG. FORCE       9.911E-05
 LARGEST RESIDUAL FORCE             4.185E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.194       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.637E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.193E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.602E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.114E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.114    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.602E-02,  FRACTION OF STEP COMPLETED  0.253    
 STEP TIME COMPLETED        2.53    ,  TOTAL TIME COMPLETED         2.53    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.114    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.601E-04   TIME AVG. FORCE       1.141E-04
 LARGEST RESIDUAL FORCE              1.08       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.290       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.758E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.852E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.140E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.099E-12   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.836E-04   TIME AVG. FORCE       1.026E-04
 LARGEST RESIDUAL FORCE             4.705E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.290       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.573E-03   AT NODE        127   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.762E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.140E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.548E-12   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.836E-04   TIME AVG. FORCE       1.026E-04
 LARGEST RESIDUAL FORCE             2.642E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.290       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.661E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.196E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.140E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.037E-12   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.171    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.114    ,  FRACTION OF STEP COMPLETED  0.265    
 STEP TIME COMPLETED        2.65    ,  TOTAL TIME COMPLETED         2.65    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.171    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 113 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.450E-04   TIME AVG. FORCE       1.323E-04
 LARGEST RESIDUAL FORCE              2.44       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.435       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.768E-02   AT NODE        541   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.646E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.711E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.391E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.033E-04   TIME AVG. FORCE       1.067E-04
 LARGEST RESIDUAL FORCE             3.135E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.435       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.910E-02   AT NODE        613   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.009E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.711E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.459E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.960E-04   TIME AVG. FORCE       1.064E-04
 LARGEST RESIDUAL FORCE             2.317E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.435       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.954E-03   AT NODE        144   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.225E-10   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.711E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.351E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.960E-04   TIME AVG. FORCE       1.064E-04
 LARGEST RESIDUAL FORCE             1.826E-09   AT NODE        164   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.435       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.692E-05   AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.431E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.711E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.276E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.257    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.171    ,  FRACTION OF STEP COMPLETED  0.282    
 STEP TIME COMPLETED        2.82    ,  TOTAL TIME COMPLETED         2.82    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.257    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 305 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.895E-02   TIME AVG. FORCE       2.370E-03
 LARGEST RESIDUAL FORCE              5.49       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.890       AT NODE        234   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.782       AT NODE        936   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.187E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.566E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.844E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 612 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      0.734       TIME AVG. FORCE       2.832E-02
 LARGEST RESIDUAL FORCE             -85.5       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         -2.09       AT NODE        715   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        -1.93       AT NODE         80   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -8.268E-02   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.566E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.645E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1140 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT PART-1-1.64 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.65 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.89 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.154 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.154 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.155 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.170 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.188 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.203 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.209 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.213 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.218 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.220 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.220 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.221 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.221 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.223 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.224 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.236 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.237 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.242 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.247 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.260 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.270 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.271 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.271 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.272 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.272 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.274 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.288 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.290 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.290 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.291 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.291 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.291 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.292 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.302 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.308 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.309 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.315 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.323 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.341 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.344 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.349 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.355 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.356 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.360 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.365 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.401 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.406 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.407 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.410 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.421 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.439 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.447 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.448 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.448 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.458 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.461 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.475 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.475 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.478 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.492 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.504 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.509 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.516 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.519 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.520 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.520 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.521 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.521 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.522 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.523 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.533 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.539 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.544 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.544 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.544 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.544 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.544 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.549 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.550 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.551 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.551 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.552 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.552 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.553 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.556 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.562 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.570 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.571 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.571 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.571 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.573 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.573 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.573 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.573 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.573 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.574 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.579 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.579 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.579 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.581 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.586 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.586 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.586 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.588 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.588 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.588 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.593 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.595 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.595 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.595 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.596 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.596 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.600 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.601 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.601 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.601 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.601 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.602 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.602 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.602 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.602 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.602 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.602 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.603 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.603 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.603 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.604 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.612 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.617 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.618 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.618 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.619 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.621 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.622 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.622 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.622 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.623 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.623 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.623 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.625 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.625 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.626 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.629 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.629 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.634 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.635 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.635 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.636 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.636 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.640 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.640 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.641 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.641 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.642 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.647 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.647 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.650 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.651 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.651 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.652 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.652 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.652 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.652 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.652 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.652 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.653 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.653 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.653 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.654 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.654 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.655 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.657 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.658 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.658 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.659 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.670 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.670 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.670 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.670 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.671 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.671 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.671 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.672 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.672 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.672 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.673 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.673 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.673 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.674 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.674 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.675 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.675 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.675 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.675 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.676 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.676 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.677 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.677 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.679 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.681 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.683 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.685 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.685 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.685 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.685 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.685 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.686 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.686 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.687 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.687 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.687 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.691 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.691 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.691 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.692 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.692 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.693 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.693 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.693 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.693 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.695 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.697 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.699 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.701 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.701 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.702 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.702 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.703 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.703 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.703 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.703 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.704 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.704 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.705 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.705 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.705 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.705 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.706 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.706 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.706 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.707 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.707 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.707 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.708 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.708 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.708 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.708 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.709 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.710 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.710 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.710 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.713 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.722 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.722 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.722 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.722 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.723 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.723 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.723 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.723 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.725 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.727 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.727 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.727 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.727 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.729 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.731 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.731 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.733 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.734 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.735 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.735 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.735 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.736 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.737 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.737 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.737 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.739 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.741 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.742 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.743 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.743 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.743 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.744 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.745 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.745 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.745 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.746 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.747 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.749 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.749 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.751 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.753 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.753 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.753 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.755 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.756 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.757 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.757 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.757 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.757 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.757 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.758 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.758 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.758 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.769 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.770 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.772 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.772 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.774 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.775 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.778 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.778 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.780 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.780 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.781 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.781 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.782 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.782 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.782 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.784 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.784 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.785 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.785 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.786 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.786 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.786 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.786 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.786 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.786 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.787 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.795 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.800 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.800 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.800 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.800 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.801 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.801 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.802 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.802 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.804 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.804 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.807 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.808 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.819 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.819 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.819 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.819 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.831 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.833 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.833 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.851 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.851 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.852 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.852 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.853 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 529 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    26 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  6.414E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.856E-04   TIME AVG. FORCE       1.171E-04
 LARGEST RESIDUAL FORCE             0.754       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.163       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.917E-03   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.037E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.414E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.863E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.007E-04   TIME AVG. FORCE       1.100E-04
 LARGEST RESIDUAL FORCE             4.015E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.163       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.082E-03   AT NODE         92   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.359E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.414E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.403E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.007E-04   TIME AVG. FORCE       1.100E-04
 LARGEST RESIDUAL FORCE             1.620E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.163       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        8.595E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.680E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.414E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.310E-13   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    9.622E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.414E-02,  FRACTION OF STEP COMPLETED  0.288    
 STEP TIME COMPLETED        2.88    ,  TOTAL TIME COMPLETED         2.88    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.622E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.962E-04   TIME AVG. FORCE       1.206E-04
 LARGEST RESIDUAL FORCE             0.772       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.244       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.959E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.336E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.622E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.995E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.077E-04   TIME AVG. FORCE       1.136E-04
 LARGEST RESIDUAL FORCE             4.127E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.244       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.102E-03   AT NODE        125   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            7.696E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.622E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.347E-12   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.077E-04   TIME AVG. FORCE       1.136E-04
 LARGEST RESIDUAL FORCE             1.712E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.244       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        8.953E-05   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.074E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.622E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.873E-13   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.144    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.622E-02,  FRACTION OF STEP COMPLETED  0.298    
 STEP TIME COMPLETED        2.98    ,  TOTAL TIME COMPLETED         2.98    


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.144    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 31 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.800E-04   TIME AVG. FORCE       1.338E-04
 LARGEST RESIDUAL FORCE              1.75       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.366       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.334E-02   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.164E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.443E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.137E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.309E-04   TIME AVG. FORCE       1.178E-04
 LARGEST RESIDUAL FORCE             6.780E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.366       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.514E-02   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.890E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.443E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.844E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.184E-04   TIME AVG. FORCE       1.173E-04
 LARGEST RESIDUAL FORCE             6.113E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.366       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.298E-03   AT NODE        109   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.869E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.443E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.017E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.183E-04   TIME AVG. FORCE       1.173E-04
 LARGEST RESIDUAL FORCE             1.933E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.366       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.129E-05   AT NODE        162   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.242E-10   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.443E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.765E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.216    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.144    ,  FRACTION OF STEP COMPLETED  0.312    
 STEP TIME COMPLETED        3.12    ,  TOTAL TIME COMPLETED         3.12    


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.216    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 225 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.051E-03   TIME AVG. FORCE       1.840E-04
 LARGEST RESIDUAL FORCE              3.93       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.126       AT NODE        792   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.824E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.165E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.660E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  2.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 29 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      9.824E-04   TIME AVG. FORCE       1.472E-04
 LARGEST RESIDUAL FORCE            -0.196       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.548       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.148       AT NODE        649   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.720E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.165E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.715E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 13 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.018E-03   TIME AVG. FORCE       1.829E-04
 LARGEST RESIDUAL FORCE            -0.135       AT NODE        722   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.548       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.217       AT NODE        757   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.763E-03   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.165E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.516E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 31 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.209E-02   TIME AVG. FORCE       1.220E-03
 LARGEST RESIDUAL FORCE              8.46       AT NODE        884   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.          1.44       AT NODE        865   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.         1.42       AT NODE        865   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.336E-02   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.165E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.048E-03   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    29 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  5.412E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.459E-04   TIME AVG. FORCE       1.252E-04
 LARGEST RESIDUAL FORCE             0.537       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.137       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.361E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.163E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.412E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.311E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.223E-04   TIME AVG. FORCE       1.210E-04
 LARGEST RESIDUAL FORCE             2.471E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.137       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.719E-04   AT NODE        125   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.798E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.412E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.432E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.223E-04   TIME AVG. FORCE       1.210E-04
 LARGEST RESIDUAL FORCE             5.969E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.137       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.492E-05   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.933E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.412E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.039E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    8.118E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.412E-02,  FRACTION OF STEP COMPLETED  0.318    
 STEP TIME COMPLETED        3.18    ,  TOTAL TIME COMPLETED         3.18    


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.118E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.544E-04   TIME AVG. FORCE       1.287E-04
 LARGEST RESIDUAL FORCE             0.549       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.205       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.390E-03   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.249E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.118E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.549E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.283E-04   TIME AVG. FORCE       1.245E-04
 LARGEST RESIDUAL FORCE             2.561E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.205       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.862E-04   AT NODE         92   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.002E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.118E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.346E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.283E-04   TIME AVG. FORCE       1.245E-04
 LARGEST RESIDUAL FORCE             6.415E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.205       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.685E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.087E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.118E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.608E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.122    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.118E-02,  FRACTION OF STEP COMPLETED  0.326    
 STEP TIME COMPLETED        3.26    ,  TOTAL TIME COMPLETED         3.26    


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.122    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.457E-04   TIME AVG. FORCE       1.381E-04
 LARGEST RESIDUAL FORCE              1.24       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.308       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.124E-03   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.519E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.218E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.386E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.374E-04   TIME AVG. FORCE       1.282E-04
 LARGEST RESIDUAL FORCE             3.113E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.308       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.905E-03   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.852E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.218E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.877E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.374E-04   TIME AVG. FORCE       1.282E-04
 LARGEST RESIDUAL FORCE             3.247E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.308       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.073E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -8.049E-12   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.218E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.611E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.183    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.122    ,  FRACTION OF STEP COMPLETED  0.338    
 STEP TIME COMPLETED        3.38    ,  TOTAL TIME COMPLETED         3.38    


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.183    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 142 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.608E-03   TIME AVG. FORCE       1.744E-04
 LARGEST RESIDUAL FORCE              2.81       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.466       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.217E-02   AT NODE        720   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.679E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.139E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 9 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.330E-04   TIME AVG. FORCE       1.440E-04
 LARGEST RESIDUAL FORCE             1.233E-02   AT NODE        520   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.460       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.096E-02   AT NODE        574   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.406E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.714E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.585E-04   TIME AVG. FORCE       1.323E-04
 LARGEST RESIDUAL FORCE             2.650E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.460       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.957E-02   AT NODE        520   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.927E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.889E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.530E-04   TIME AVG. FORCE       1.321E-04
 LARGEST RESIDUAL FORCE            -9.507E-04   AT NODE        521   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.460       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.396E-02   AT NODE        378   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.841E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.310E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.511E-04   TIME AVG. FORCE       1.320E-04
 LARGEST RESIDUAL FORCE             5.340E-06   AT NODE        540   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.460       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.035E-03   AT NODE        522   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.595E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.789E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      2.511E-04   TIME AVG. FORCE       1.320E-04
 LARGEST RESIDUAL FORCE             1.478E-08   AT NODE        522   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.460       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.022E-05   AT NODE        522   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.336E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.153E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.183    ,  FRACTION OF STEP COMPLETED  0.356    
 STEP TIME COMPLETED        3.56    ,  TOTAL TIME COMPLETED         3.56    


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.183    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 185 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.169E-03   TIME AVG. FORCE       1.635E-04
 LARGEST RESIDUAL FORCE              3.34       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.460       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.434E-02   AT NODE        591   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.186E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.084E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.053E-04   TIME AVG. FORCE       1.373E-04
 LARGEST RESIDUAL FORCE            -8.539E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.470       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.612E-02   AT NODE        591   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.121E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.170E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  7.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  7.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.650E-04   TIME AVG. FORCE       1.361E-04
 LARGEST RESIDUAL FORCE            -3.975E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.459       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.466E-02   AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.706E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.322E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.649E-04   TIME AVG. FORCE       1.361E-04
 LARGEST RESIDUAL FORCE             7.276E-06   AT NODE        630   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.459       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.368E-03   AT NODE        594   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.683E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.879E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      2.649E-04   TIME AVG. FORCE       1.361E-04
 LARGEST RESIDUAL FORCE             7.475E-10   AT NODE        558   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.459       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.465E-05   AT NODE        559   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.302E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.724E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.183    ,  FRACTION OF STEP COMPLETED  0.374    
 STEP TIME COMPLETED        3.74    ,  TOTAL TIME COMPLETED         3.74    


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.183    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 185 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.562E-03   TIME AVG. FORCE       2.074E-04
 LARGEST RESIDUAL FORCE              3.36       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.458       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.151       AT NODE        591   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.748E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.249E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 36 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.106E-03   TIME AVG. FORCE       3.705E-04
 LARGEST RESIDUAL FORCE             -4.98       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.576       AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.813       AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -8.686E-03   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.248E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 131 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      0.101       TIME AVG. FORCE       3.106E-03
 LARGEST RESIDUAL FORCE              10.1       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.          1.44       AT NODE        649   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.         1.45       AT NODE        649   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            -8.68       AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.827E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.465E-04   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 484 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT PART-1-1.48 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.49 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.82 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.151 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.556 IS DISTORTING EXCESSIVELY.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 5 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    34 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  4.567E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.817E-04   TIME AVG. FORCE       1.433E-04
 LARGEST RESIDUAL FORCE             0.522       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.115       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.309E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.281E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.567E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.146E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.684E-04   TIME AVG. FORCE       1.400E-04
 LARGEST RESIDUAL FORCE             2.360E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.115       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.371E-04   AT NODE         74   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.890E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.567E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.383E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.684E-04   TIME AVG. FORCE       1.400E-04
 LARGEST RESIDUAL FORCE             5.405E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.115       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.244E-05   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.016E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.567E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.191E-12   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.567E-02,  FRACTION OF STEP COMPLETED  0.379    
 STEP TIME COMPLETED        3.79    ,  TOTAL TIME COMPLETED         3.79    


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.567E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.044E-04   TIME AVG. FORCE       1.446E-04
 LARGEST RESIDUAL FORCE             0.209       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.114       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.229E-04   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.361E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.567E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.817E-13   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.719E-04   TIME AVG. FORCE       1.437E-04
 LARGEST RESIDUAL FORCE             4.630E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.114       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.028E-04   AT NODE        143   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.496E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.567E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.700E-12   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.719E-04   TIME AVG. FORCE       1.437E-04
 LARGEST RESIDUAL FORCE             2.037E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.114       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.094E-06   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.673E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.567E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.414E-12   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    6.850E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.567E-02,  FRACTION OF STEP COMPLETED  0.384    
 STEP TIME COMPLETED        3.84    ,  TOTAL TIME COMPLETED         3.84    


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.850E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.549E-04   TIME AVG. FORCE       1.496E-04
 LARGEST RESIDUAL FORCE             0.391       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.172       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -9.795E-04   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            7.270E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.850E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.606E-13   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.771E-04   TIME AVG. FORCE       1.474E-04
 LARGEST RESIDUAL FORCE             1.453E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.172       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.557E-04   AT NODE         74   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.200E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.850E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.613E-13   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.772E-04   TIME AVG. FORCE       1.474E-04
 LARGEST RESIDUAL FORCE             2.022E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.172       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.426E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.466E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.850E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.983E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.103    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.850E-02,  FRACTION OF STEP COMPLETED  0.390    
 STEP TIME COMPLETED        3.90    ,  TOTAL TIME COMPLETED         3.90    


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.103    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.906E-04   TIME AVG. FORCE       1.567E-04
 LARGEST RESIDUAL FORCE             0.880       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.257       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.201E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.695E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.027E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.372E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.851E-04   TIME AVG. FORCE       1.511E-04
 LARGEST RESIDUAL FORCE             4.778E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.257       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.227E-03   AT NODE        142   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            7.241E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.027E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.651E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.850E-04   TIME AVG. FORCE       1.511E-04
 LARGEST RESIDUAL FORCE             2.292E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.257       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.145E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.260E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.027E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.925E-13   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.154    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.103    ,  FRACTION OF STEP COMPLETED  0.401    
 STEP TIME COMPLETED        4.01    ,  TOTAL TIME COMPLETED         4.01    


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.154    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 57 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.971E-04   TIME AVG. FORCE       1.681E-04
 LARGEST RESIDUAL FORCE              1.98       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.399       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.493E-02   AT NODE        451   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.162E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.541E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.136E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.044E-04   TIME AVG. FORCE       1.552E-04
 LARGEST RESIDUAL FORCE             2.467E-03   AT NODE        379   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.385       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.176E-02   AT NODE        451   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.839E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.541E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.625E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.970E-04   TIME AVG. FORCE       1.550E-04
 LARGEST RESIDUAL FORCE             2.235E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.384       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.189E-03   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.020E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.541E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.917E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      2.970E-04   TIME AVG. FORCE       1.550E-04
 LARGEST RESIDUAL FORCE             8.390E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.384       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.594E-05   AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.118E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.541E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.192E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.231    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.154    ,  FRACTION OF STEP COMPLETED  0.416    
 STEP TIME COMPLETED        4.16    ,  TOTAL TIME COMPLETED         4.16    


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.231    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 254 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.821E-03   TIME AVG. FORCE       1.977E-04
 LARGEST RESIDUAL FORCE              4.47       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.583       AT NODE        162   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.465E-02   AT NODE        722   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.452E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.640E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.844E-04   TIME AVG. FORCE       1.660E-04
 LARGEST RESIDUAL FORCE            -1.942E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.576       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.878E-02   AT NODE        504   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            7.916E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.979E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.731E-04   TIME AVG. FORCE       1.606E-04
 LARGEST RESIDUAL FORCE            -1.037E-02   AT NODE        809   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.574       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.498E-02   AT NODE        810   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.456E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.776E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.186E-04   TIME AVG. FORCE       1.592E-04
 LARGEST RESIDUAL FORCE             1.329E-03   AT NODE        846   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.574       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.088E-02   AT NODE        864   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.351E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.471E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.161E-04   TIME AVG. FORCE       1.591E-04
 LARGEST RESIDUAL FORCE             5.258E-04   AT NODE        738   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.574       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.457E-02   AT NODE        810   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.911E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.589E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      3.151E-04   TIME AVG. FORCE       1.591E-04
 LARGEST RESIDUAL FORCE             4.682E-06   AT NODE        846   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.574       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.376E-03   AT NODE        738   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.522E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.783E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      3.151E-04   TIME AVG. FORCE       1.591E-04
 LARGEST RESIDUAL FORCE             9.482E-09   AT NODE        738   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.574       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.632E-05   AT NODE        738   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.137E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.926E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   7 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  7 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.231    ,  FRACTION OF STEP COMPLETED  0.439    
 STEP TIME COMPLETED        4.39    ,  TOTAL TIME COMPLETED         4.39    


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.231    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 293 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.752E-03   TIME AVG. FORCE       1.989E-04
 LARGEST RESIDUAL FORCE              5.34       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.574       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.414E-02   AT NODE        793   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.130E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.605E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 7 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.857E-04   TIME AVG. FORCE       1.648E-04
 LARGEST RESIDUAL FORCE            -8.137E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.571       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.294E-02   AT NODE        794   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.109E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.019E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.331E-04   TIME AVG. FORCE       1.634E-04
 LARGEST RESIDUAL FORCE            -5.351E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.571       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.375E-02   AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.906E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.106E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  6.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.335E-04   TIME AVG. FORCE       1.634E-04
 LARGEST RESIDUAL FORCE            -1.982E-04   AT NODE        811   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.571       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.409E-03   AT NODE        793   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.354E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.957E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.334E-04   TIME AVG. FORCE       1.634E-04
 LARGEST RESIDUAL FORCE            -4.713E-07   AT NODE        811   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.571       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.288E-04   AT NODE        793   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.942E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.085E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.231    ,  FRACTION OF STEP COMPLETED  0.462    
 STEP TIME COMPLETED        4.62    ,  TOTAL TIME COMPLETED         4.62    


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.231    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 293 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.816E-03   TIME AVG. FORCE       2.038E-04
 LARGEST RESIDUAL FORCE              5.35       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.570       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.293E-02   AT NODE        865   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.384E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.611E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 9 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.302E-04   TIME AVG. FORCE       1.700E-04
 LARGEST RESIDUAL FORCE             6.636E-03   AT NODE        809   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.072E-02   AT NODE        809   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.392E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.295E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.655E-04   TIME AVG. FORCE       1.684E-04
 LARGEST RESIDUAL FORCE             4.518E-03   AT NODE        830   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.227E-02   AT NODE        865   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.580E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.496E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.524E-04   TIME AVG. FORCE       1.681E-04
 LARGEST RESIDUAL FORCE            -3.296E-04   AT NODE        827   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.567       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        9.554E-03   AT NODE        810   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.169E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.405E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.519E-04   TIME AVG. FORCE       1.680E-04
 LARGEST RESIDUAL FORCE             3.078E-05   AT NODE        830   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.567       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.537E-03   AT NODE        847   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.500E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.034E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      3.519E-04   TIME AVG. FORCE       1.680E-04
 LARGEST RESIDUAL FORCE            -1.701E-08   AT NODE        847   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.567       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.694E-05   AT NODE        847   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.997E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.741E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.231    ,  FRACTION OF STEP COMPLETED  0.485    
 STEP TIME COMPLETED        4.85    ,  TOTAL TIME COMPLETED         4.85    


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.231    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 292 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.735E-03   TIME AVG. FORCE       2.292E-04
 LARGEST RESIDUAL FORCE              5.36       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.564       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.154       AT NODE        865   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.138E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.252E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 37 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.278E-02   TIME AVG. FORCE       1.183E-03
 LARGEST RESIDUAL FORCE             -5.64       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         -1.01       AT NODE        738   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.805       AT NODE        882   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.416E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.448E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 516 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.918E-02   TIME AVG. FORCE       1.097E-03
 LARGEST RESIDUAL FORCE              7.63       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         -1.08       AT NODE        594   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.855       AT NODE       1152   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.185E-02   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.312E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.120E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 574 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT PART-1-1.17 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.35 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.52 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.67 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.67 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.67 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.116 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.137 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.137 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.140 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.170 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.188 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.207 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.253 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.254 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.262 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.264 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.264 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.264 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.265 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.265 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.267 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.272 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.290 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.295 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.298 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.298 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.298 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.300 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.307 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.313 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.313 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.314 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.315 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.315 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.324 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.331 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.331 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.331 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.340 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.346 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.346 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.346 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.346 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.347 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.347 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.347 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.347 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.347 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.349 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.349 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.350 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.351 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.351 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.361 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.364 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.364 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.366 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.366 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.366 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.368 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.368 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.369 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.370 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.370 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.376 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.381 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.382 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.382 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.382 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.382 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.382 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.390 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.391 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.396 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.396 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.397 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.398 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.407 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.408 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.417 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.424 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.424 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.424 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.426 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.432 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.432 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.432 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.436 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.436 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.436 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.441 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.442 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.448 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.448 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.448 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.448 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.451 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.463 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.469 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.469 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.470 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.478 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.480 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.480 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.480 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.481 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.481 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.481 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.481 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.483 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.483 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.484 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.484 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.485 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.485 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.485 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.485 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.486 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.486 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.487 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.487 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.489 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.489 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.489 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.489 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.490 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.490 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.490 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.492 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.498 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.498 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.498 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.498 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.500 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.501 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.501 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.507 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.514 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.515 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.524 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.529 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.543 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.548 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.557 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.558 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.561 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.562 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.575 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.576 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.580 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.580 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.580 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.582 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.592 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.594 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.594 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.594 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.596 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.596 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.596 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.596 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.598 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.599 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.610 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.610 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.610 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.610 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.612 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.614 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.626 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.626 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.633 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.633 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.633 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.633 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.643 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.643 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.643 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.643 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.650 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.661 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.661 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.661 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.661 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.662 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.769 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 331 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    42 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  5.780E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.383E-04   TIME AVG. FORCE       1.769E-04
 LARGEST RESIDUAL FORCE             0.836       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.141       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.045E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.357E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.780E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.510E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.566E-04   TIME AVG. FORCE       1.725E-04
 LARGEST RESIDUAL FORCE             4.616E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.141       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.135E-03   AT NODE        142   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.291E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.780E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.427E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.566E-04   TIME AVG. FORCE       1.725E-04
 LARGEST RESIDUAL FORCE             2.088E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.141       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.041E-04   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.425E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.780E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.948E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.780E-02,  FRACTION OF STEP COMPLETED  0.491    
 STEP TIME COMPLETED        4.91    ,  TOTAL TIME COMPLETED         4.91    


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.780E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.158E-04   TIME AVG. FORCE       1.782E-04
 LARGEST RESIDUAL FORCE             0.335       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.141       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.151E-04   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.476E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.780E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.088E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.613E-04   TIME AVG. FORCE       1.769E-04
 LARGEST RESIDUAL FORCE             1.099E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.141       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.564E-04   AT NODE        143   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.697E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.780E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.114E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.613E-04   TIME AVG. FORCE       1.769E-04
 LARGEST RESIDUAL FORCE             1.136E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.141       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.782E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.073E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.780E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.298E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    8.669E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.780E-02,  FRACTION OF STEP COMPLETED  0.497    
 STEP TIME COMPLETED        4.97    ,  TOTAL TIME COMPLETED         4.97    


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.669E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.947E-04   TIME AVG. FORCE       1.841E-04
 LARGEST RESIDUAL FORCE             0.627       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.211       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.525E-03   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.059E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.669E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.980E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.683E-04   TIME AVG. FORCE       1.813E-04
 LARGEST RESIDUAL FORCE             3.162E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.211       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.345E-04   AT NODE         74   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.557E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.669E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.688E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.683E-04   TIME AVG. FORCE       1.813E-04
 LARGEST RESIDUAL FORCE             9.589E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.211       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.011E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.247E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.669E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.280E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.130    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.669E-02,  FRACTION OF STEP COMPLETED  0.506    
 STEP TIME COMPLETED        5.06    ,  TOTAL TIME COMPLETED         5.06    


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.130    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.031E-04   TIME AVG. FORCE       1.929E-04
 LARGEST RESIDUAL FORCE              1.41       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.316       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.825E-03   AT NODE        234   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.445E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.300E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.519E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      3.791E-04   TIME AVG. FORCE       1.857E-04
 LARGEST RESIDUAL FORCE             8.470E-05   AT NODE        233   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.315       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.696E-03   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.214E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.300E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.525E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.790E-04   TIME AVG. FORCE       1.857E-04
 LARGEST RESIDUAL FORCE             3.897E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.315       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.318E-04   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.409E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.300E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.677E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.790E-04   TIME AVG. FORCE       1.857E-04
 LARGEST RESIDUAL FORCE            -1.255E-11   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.315       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.627E-07   AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            7.689E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.300E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.701E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.195    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.130    ,  FRACTION OF STEP COMPLETED  0.519    
 STEP TIME COMPLETED        5.19    ,  TOTAL TIME COMPLETED         5.19    


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.195    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 169 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.175E-03   TIME AVG. FORCE       2.072E-04
 LARGEST RESIDUAL FORCE              3.17       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.475       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.560E-02   AT NODE        563   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.497E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.766E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.081E-04   TIME AVG. FORCE       1.905E-04
 LARGEST RESIDUAL FORCE            -5.444E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.482       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.532E-02   AT NODE        306   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.942E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.922E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.949E-04   TIME AVG. FORCE       1.902E-04
 LARGEST RESIDUAL FORCE            -2.030E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.470       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.245E-02   AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.993E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.306E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      3.951E-04   TIME AVG. FORCE       1.902E-04
 LARGEST RESIDUAL FORCE             1.239E-06   AT NODE        630   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.470       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.696E-04   AT NODE        522   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.228E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.253E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      3.951E-04   TIME AVG. FORCE       1.902E-04
 LARGEST RESIDUAL FORCE            -2.428E-11   AT NODE        379   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.470       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.482E-06   AT NODE        450   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.000E-11   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.365E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.195    ,  FRACTION OF STEP COMPLETED  0.538    
 STEP TIME COMPLETED        5.38    ,  TOTAL TIME COMPLETED         5.38    


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.195    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 209 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.332E-03   TIME AVG. FORCE       2.145E-04
 LARGEST RESIDUAL FORCE              3.81       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.474       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.085E-02   AT NODE        774   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.084E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.366E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.150E-04   TIME AVG. FORCE       1.950E-04
 LARGEST RESIDUAL FORCE            -2.436E-03   AT NODE        774   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.468       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.165E-02   AT NODE        774   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.857E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.329E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.115E-04   TIME AVG. FORCE       1.949E-04
 LARGEST RESIDUAL FORCE            -6.507E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.467       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -9.954E-03   AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.028E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.730E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.113E-04   TIME AVG. FORCE       1.949E-04
 LARGEST RESIDUAL FORCE            -9.010E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.467       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.025E-04   AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.522E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.380E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.195    ,  FRACTION OF STEP COMPLETED  0.558    
 STEP TIME COMPLETED        5.58    ,  TOTAL TIME COMPLETED         5.58    


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.195    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 208 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.338E-03   TIME AVG. FORCE       2.187E-04
 LARGEST RESIDUAL FORCE              3.81       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.471       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.403E-02   AT NODE        792   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.123E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.170E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.297E-04   TIME AVG. FORCE       1.998E-04
 LARGEST RESIDUAL FORCE            -1.223E-03   AT NODE        774   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.464       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.600E-02   AT NODE        200   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.874E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.211E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.281E-04   TIME AVG. FORCE       1.998E-04
 LARGEST RESIDUAL FORCE            -1.081E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.464       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.300E-02   AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.529E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.235E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.278E-04   TIME AVG. FORCE       1.998E-04
 LARGEST RESIDUAL FORCE            -3.528E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.464       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.248E-04   AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -8.108E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.951E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.340E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.293    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.195    ,  FRACTION OF STEP COMPLETED  0.577    
 STEP TIME COMPLETED        5.77    ,  TOTAL TIME COMPLETED         5.77    


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.293    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 344 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.248E-03   TIME AVG. FORCE       2.416E-04
 LARGEST RESIDUAL FORCE              7.13       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.695       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.435E-02   AT NODE       1008   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.127E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.349E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 9 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.334E-04   TIME AVG. FORCE       2.066E-04
 LARGEST RESIDUAL FORCE            -4.573E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.689       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.099E-02   AT NODE        937   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.486E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.758E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.611E-04   TIME AVG. FORCE       2.051E-04
 LARGEST RESIDUAL FORCE            -2.495E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.689       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.799E-02   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.245E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.538E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.531E-04   TIME AVG. FORCE       2.049E-04
 LARGEST RESIDUAL FORCE             5.148E-04   AT NODE        918   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.689       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.088E-02   AT NODE        936   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            7.592E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.863E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.529E-04   TIME AVG. FORCE       2.049E-04
 LARGEST RESIDUAL FORCE             1.905E-05   AT NODE        918   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.690       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.977E-03   AT NODE        936   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.469E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.022E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      4.529E-04   TIME AVG. FORCE       2.049E-04
 LARGEST RESIDUAL FORCE             3.297E-08   AT NODE        918   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.690       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        8.251E-05   AT NODE        936   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.810E-12   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.997E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.293    ,  FRACTION OF STEP COMPLETED  0.606    
 STEP TIME COMPLETED        6.06    ,  TOTAL TIME COMPLETED         6.06    


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.293    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 380 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.737E-03   TIME AVG. FORCE       2.556E-04
 LARGEST RESIDUAL FORCE              8.55       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.680       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.100       AT NODE        559   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.450E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.884E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 14 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.704E-04   TIME AVG. FORCE       2.142E-04
 LARGEST RESIDUAL FORCE            -9.747E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.685       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -9.631E-02   AT NODE        522   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.176E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.102E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.266E-04   TIME AVG. FORCE       2.114E-04
 LARGEST RESIDUAL FORCE            -5.411E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.701       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.447E-02   AT NODE        432   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.220E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.792E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      4.876E-04   TIME AVG. FORCE       2.106E-04
 LARGEST RESIDUAL FORCE             2.034E-03   AT NODE        486   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.685       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.681E-02   AT NODE        450   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.184E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.133E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      4.794E-04   TIME AVG. FORCE       2.104E-04
 LARGEST RESIDUAL FORCE             5.971E-04   AT NODE        486   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.682       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.393E-02   AT NODE        504   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.754E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.066E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      4.785E-04   TIME AVG. FORCE       2.104E-04
 LARGEST RESIDUAL FORCE             1.928E-05   AT NODE        486   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.682       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.018E-03   AT NODE        450   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.486E-09   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.925E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      4.784E-04   TIME AVG. FORCE       2.104E-04
 LARGEST RESIDUAL FORCE             5.058E-08   AT NODE        486   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.682       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.274E-04   AT NODE        504   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.621E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.399E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   7 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  7 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.293    ,  FRACTION OF STEP COMPLETED  0.636    
 STEP TIME COMPLETED        6.36    ,  TOTAL TIME COMPLETED         6.36    


  INCREMENT    51 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.293    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 379 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.125E-03   TIME AVG. FORCE       2.676E-04
 LARGEST RESIDUAL FORCE              8.55       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.671       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.157       AT NODE       1008   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.911E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.356E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 32 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.239E-03   TIME AVG. FORCE       2.894E-04
 LARGEST RESIDUAL FORCE            -0.550       AT NODE        989   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.753       AT NODE        378   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.394       AT NODE       1080   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.899E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.013E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 115 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      3.683E-03   TIME AVG. FORCE       2.785E-04
 LARGEST RESIDUAL FORCE              1.46       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.965       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.274       AT NODE        162   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.384E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.076E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 90 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.333E-02   TIME AVG. FORCE       4.677E-04
 LARGEST RESIDUAL FORCE             -1.92       AT NODE       1063   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.957       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.663       AT NODE       1063   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            0.275       AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.926E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.687E-04   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    51 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  7.315E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.657E-04   TIME AVG. FORCE       2.213E-04
 LARGEST RESIDUAL FORCE              1.34       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.170       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.659E-03   AT NODE        271   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.132E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.315E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.630E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  2.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  7.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.850E-04   TIME AVG. FORCE       2.158E-04
 LARGEST RESIDUAL FORCE             3.000E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.169       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.466E-03   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.700E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.315E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.334E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.848E-04   TIME AVG. FORCE       2.158E-04
 LARGEST RESIDUAL FORCE             6.339E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.169       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.523E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.971E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.315E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.920E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.315E-02,  FRACTION OF STEP COMPLETED  0.643    
 STEP TIME COMPLETED        6.43    ,  TOTAL TIME COMPLETED         6.43    


  INCREMENT    52 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.315E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.791E-04   TIME AVG. FORCE       2.228E-04
 LARGEST RESIDUAL FORCE             0.536       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.169       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.231E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            9.131E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.315E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.426E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      4.913E-04   TIME AVG. FORCE       2.211E-04
 LARGEST RESIDUAL FORCE             2.485E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.169       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.597E-04   AT NODE        143   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.963E-10   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.315E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.391E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      4.913E-04   TIME AVG. FORCE       2.211E-04
 LARGEST RESIDUAL FORCE             5.738E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.169       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.409E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.356E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.315E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.532E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.110    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.315E-02,  FRACTION OF STEP COMPLETED  0.650    
 STEP TIME COMPLETED        6.50    ,  TOTAL TIME COMPLETED         6.50    


  INCREMENT    53 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.110    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.978E-04   TIME AVG. FORCE       2.301E-04
 LARGEST RESIDUAL FORCE              1.01       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.253       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.298E-03   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.134E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.097E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.899E-11   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.010E-04   TIME AVG. FORCE       2.264E-04
 LARGEST RESIDUAL FORCE             5.585E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.252       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.290E-03   AT NODE         75   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.051E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.097E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.423E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.010E-04   TIME AVG. FORCE       2.264E-04
 LARGEST RESIDUAL FORCE             2.990E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.252       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.460E-04   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.818E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.097E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.545E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.165    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.110    ,  FRACTION OF STEP COMPLETED  0.661    
 STEP TIME COMPLETED        6.61    ,  TOTAL TIME COMPLETED         6.61    


  INCREMENT    54 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.165    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 78 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.178E-03   TIME AVG. FORCE       2.440E-04
 LARGEST RESIDUAL FORCE              2.33       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.374       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.548E-02   AT NODE        417   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.425E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.381E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.272E-04   TIME AVG. FORCE       2.338E-04
 LARGEST RESIDUAL FORCE             2.998E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.374       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.593E-02   AT NODE        417   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.676E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.408E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.169E-04   TIME AVG. FORCE       2.317E-04
 LARGEST RESIDUAL FORCE             5.861E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.374       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.206E-02   AT NODE        415   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.255E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.117E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.164E-04   TIME AVG. FORCE       2.317E-04
 LARGEST RESIDUAL FORCE             3.124E-04   AT NODE        488   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.375       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.518E-02   AT NODE        487   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.317E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.347E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.158E-04   TIME AVG. FORCE       2.317E-04
 LARGEST RESIDUAL FORCE            -8.560E-07   AT NODE        379   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.375       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.433E-04   AT NODE        415   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -8.368E-10   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.789E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.165    ,  FRACTION OF STEP COMPLETED  0.678    
 STEP TIME COMPLETED        6.78    ,  TOTAL TIME COMPLETED         6.78    


  INCREMENT    55 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.165    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 122 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.735E-03   TIME AVG. FORCE       2.591E-04
 LARGEST RESIDUAL FORCE              2.74       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.452       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.112       AT NODE        540   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.576E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.059E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 28 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.599E-04   TIME AVG. FORCE       2.413E-04
 LARGEST RESIDUAL FORCE             2.130E-02   AT NODE        468   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.395       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        9.190E-02   AT NODE        415   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.173E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.485E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.090E-03   TIME AVG. FORCE       2.473E-04
 LARGEST RESIDUAL FORCE             3.392E-02   AT NODE        467   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.371       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.186       AT NODE        576   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.020E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.911E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.016E-04   TIME AVG. FORCE       2.385E-04
 LARGEST RESIDUAL FORCE             3.038E-02   AT NODE        541   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.371       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.162E-02   AT NODE        488   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.824E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.956E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      9.598E-04   TIME AVG. FORCE       2.450E-04
 LARGEST RESIDUAL FORCE             0.155       AT NODE        449   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.376       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.180       AT NODE        450   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.511E-03   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.646E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.612E-04   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    55 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  4.115E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.800E-04   TIME AVG. FORCE       2.381E-04
 LARGEST RESIDUAL FORCE             0.424       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -9.379E-02   AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -9.589E-04   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.964E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.115E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.197E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.195E-04   TIME AVG. FORCE       2.370E-04
 LARGEST RESIDUAL FORCE             1.672E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -9.336E-02   AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.165E-04   AT NODE        143   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.613E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.115E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.745E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.195E-04   TIME AVG. FORCE       2.370E-04
 LARGEST RESIDUAL FORCE             2.573E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         9.336E-02   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.806E-05   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.027E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.115E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.426E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.115E-02,  FRACTION OF STEP COMPLETED  0.682    
 STEP TIME COMPLETED        6.82    ,  TOTAL TIME COMPLETED         6.82    


  INCREMENT    56 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.115E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.373E-04   TIME AVG. FORCE       2.423E-04
 LARGEST RESIDUAL FORCE             0.170       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         9.335E-02   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.819E-04   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.720E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.115E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.800E-11   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.233E-04   TIME AVG. FORCE       2.421E-04
 LARGEST RESIDUAL FORCE             3.089E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         9.317E-02   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.072E-04   AT NODE         74   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.118E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.115E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.250E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.233E-04   TIME AVG. FORCE       2.421E-04
 LARGEST RESIDUAL FORCE             8.593E-10   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -9.317E-02   AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.732E-06   AT NODE        109   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.642E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      4.115E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.715E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    6.172E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.115E-02,  FRACTION OF STEP COMPLETED  0.686    
 STEP TIME COMPLETED        6.86    ,  TOTAL TIME COMPLETED         6.86    


  INCREMENT    57 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.172E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.671E-04   TIME AVG. FORCE       2.478E-04
 LARGEST RESIDUAL FORCE             0.318       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.140       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.143E-04   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.697E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.172E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.308E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.288E-04   TIME AVG. FORCE       2.471E-04
 LARGEST RESIDUAL FORCE             1.000E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.139       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.844E-04   AT NODE        143   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.095E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.172E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.474E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.288E-04   TIME AVG. FORCE       2.471E-04
 LARGEST RESIDUAL FORCE             9.115E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.139       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.603E-05   AT NODE        108   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.252E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.172E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.828E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    9.258E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.172E-02,  FRACTION OF STEP COMPLETED  0.692    
 STEP TIME COMPLETED        6.92    ,  TOTAL TIME COMPLETED         6.92    


  INCREMENT    58 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.258E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.609E-04   TIME AVG. FORCE       2.542E-04
 LARGEST RESIDUAL FORCE             0.716       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.209       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.602E-03   AT NODE         55   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.718E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.258E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.979E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.373E-04   TIME AVG. FORCE       2.521E-04
 LARGEST RESIDUAL FORCE             3.895E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.208       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.651E-04   AT NODE         75   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.480E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.258E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.961E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.373E-04   TIME AVG. FORCE       2.521E-04
 LARGEST RESIDUAL FORCE             1.405E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.208       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.674E-05   AT NODE        108   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.198E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.258E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.929E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.139    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.258E-02,  FRACTION OF STEP COMPLETED  0.701    
 STEP TIME COMPLETED        7.01    ,  TOTAL TIME COMPLETED         7.01    


  INCREMENT    59 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.139    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.028E-04   TIME AVG. FORCE       2.631E-04
 LARGEST RESIDUAL FORCE              1.62       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.309       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.512E-02   AT NODE        271   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.232E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.316E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.610E-04   TIME AVG. FORCE       2.573E-04
 LARGEST RESIDUAL FORCE             6.204E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.309       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.967E-02   AT NODE        162   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.568E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.135E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.503E-04   TIME AVG. FORCE       2.572E-04
 LARGEST RESIDUAL FORCE            -5.896E-04   AT NODE        145   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.312       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.943E-02   AT NODE         73   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.401E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.750E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.501E-04   TIME AVG. FORCE       2.572E-04
 LARGEST RESIDUAL FORCE            -6.363E-05   AT NODE        145   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.311       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.038E-03   AT NODE        271   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.512E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.360E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.501E-04   TIME AVG. FORCE       2.572E-04
 LARGEST RESIDUAL FORCE             7.149E-08   AT NODE        146   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.311       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.384E-04   AT NODE        145   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.313E-10   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.291E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.139    ,  FRACTION OF STEP COMPLETED  0.715    
 STEP TIME COMPLETED        7.15    ,  TOTAL TIME COMPLETED         7.15    


  INCREMENT    60 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.139    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 37 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.679E-04   TIME AVG. FORCE       2.690E-04
 LARGEST RESIDUAL FORCE              1.94       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.306       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.173E-02   AT NODE        151   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.109E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.626E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.641E-04   TIME AVG. FORCE       2.623E-04
 LARGEST RESIDUAL FORCE             3.354E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.308       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.290E-02   AT NODE        151   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.529E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.645E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.629E-04   TIME AVG. FORCE       2.623E-04
 LARGEST RESIDUAL FORCE             2.921E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.308       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.558E-03   AT NODE         80   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.018E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.353E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.630E-04   TIME AVG. FORCE       2.623E-04
 LARGEST RESIDUAL FORCE             1.575E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.308       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.574E-06   AT NODE        151   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.798E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.550E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.139    ,  FRACTION OF STEP COMPLETED  0.729    
 STEP TIME COMPLETED        7.29    ,  TOTAL TIME COMPLETED         7.29    


  INCREMENT    61 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.139    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 37 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.777E-04   TIME AVG. FORCE       2.740E-04
 LARGEST RESIDUAL FORCE              1.94       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.304       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.101E-02   AT NODE        210   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.657E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.871E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      5.765E-04   TIME AVG. FORCE       2.674E-04
 LARGEST RESIDUAL FORCE             3.164E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.306       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.208E-02   AT NODE        210   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.251E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.230E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      5.762E-04   TIME AVG. FORCE       2.674E-04
 LARGEST RESIDUAL FORCE             2.676E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.306       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.732E-03   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.251E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.131E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.762E-04   TIME AVG. FORCE       2.674E-04
 LARGEST RESIDUAL FORCE             1.181E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.306       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.361E-06   AT NODE        210   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.655E-10   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.389E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.796E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.208    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.139    ,  FRACTION OF STEP COMPLETED  0.743    
 STEP TIME COMPLETED        7.43    ,  TOTAL TIME COMPLETED         7.43    


  INCREMENT    62 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.208    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 184 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.475E-03   TIME AVG. FORCE       2.869E-04
 LARGEST RESIDUAL FORCE              3.64       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.465       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.412E-02   AT NODE        649   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.835E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.793E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.559E-04   TIME AVG. FORCE       2.737E-04
 LARGEST RESIDUAL FORCE            -3.123E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.451       AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.665E-02   AT NODE        720   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.678E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.607E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.329E-04   TIME AVG. FORCE       2.733E-04
 LARGEST RESIDUAL FORCE            -2.129E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.453       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.207E-02   AT NODE        648   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.379E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.453E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      5.973E-04   TIME AVG. FORCE       2.727E-04
 LARGEST RESIDUAL FORCE             8.449E-04   AT NODE        702   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.454       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.605E-02   AT NODE        720   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.671E-09   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.617E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      5.963E-04   TIME AVG. FORCE       2.727E-04
 LARGEST RESIDUAL FORCE             2.561E-04   AT NODE        630   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.454       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.559E-03   AT NODE        648   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.680E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.079E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      5.958E-04   TIME AVG. FORCE       2.727E-04
 LARGEST RESIDUAL FORCE             9.182E-07   AT NODE        702   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.454       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.752E-04   AT NODE        648   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.311E-11   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.568E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.208    ,  FRACTION OF STEP COMPLETED  0.764    
 STEP TIME COMPLETED        7.64    ,  TOTAL TIME COMPLETED         7.64    


  INCREMENT    63 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.208    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 224 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.012E-03   TIME AVG. FORCE       3.003E-04
 LARGEST RESIDUAL FORCE              4.34       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.457       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        8.487E-02   AT NODE        589   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.687E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.534E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 11 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.357E-04   TIME AVG. FORCE       2.816E-04
 LARGEST RESIDUAL FORCE            -4.083E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.445       AT NODE         55   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.628E-02   AT NODE        705   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.533E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.616E-10   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.388E-04   TIME AVG. FORCE       2.785E-04
 LARGEST RESIDUAL FORCE            -1.134E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.447       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.201E-02   AT NODE        649   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.834E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.280E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.183E-04   TIME AVG. FORCE       2.782E-04
 LARGEST RESIDUAL FORCE            -6.558E-04   AT NODE        665   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.449       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.482E-02   AT NODE        810   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.591E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.283E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      6.160E-04   TIME AVG. FORCE       2.781E-04
 LARGEST RESIDUAL FORCE             3.211E-05   AT NODE        666   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.449       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.164E-03   AT NODE        738   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.256E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.086E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      6.158E-04   TIME AVG. FORCE       2.781E-04
 LARGEST RESIDUAL FORCE             4.591E-08   AT NODE        738   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.449       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.031E-04   AT NODE        738   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.769E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.295E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.208    ,  FRACTION OF STEP COMPLETED  0.785    
 STEP TIME COMPLETED        7.85    ,  TOTAL TIME COMPLETED         7.85    


  INCREMENT    64 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.208    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 223 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.847E-03   TIME AVG. FORCE       3.027E-04
 LARGEST RESIDUAL FORCE              4.37       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.451       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.511E-02   AT NODE        589   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            9.042E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.093E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.717E-04   TIME AVG. FORCE       2.874E-04
 LARGEST RESIDUAL FORCE            -7.100E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.518       AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.138       AT NODE        487   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.351E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.457E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      1.610E-03   TIME AVG. FORCE       2.990E-04
 LARGEST RESIDUAL FORCE             0.481       AT NODE        791   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.466       AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.243       AT NODE        792   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.827E-02   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.667E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 24 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      1.490E-02   TIME AVG. FORCE       5.065E-04
 LARGEST RESIDUAL FORCE              7.79       AT NODE        740   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.          1.52       AT NODE        740   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.         1.54       AT NODE        740   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.933E-02   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.083E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.976E-04   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    64 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  5.207E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.292E-04   TIME AVG. FORCE       2.852E-04
 LARGEST RESIDUAL FORCE             0.680       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.112       AT NODE         90   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.444E-03   AT NODE         18   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.434E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.207E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.683E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.210E-04   TIME AVG. FORCE       2.835E-04
 LARGEST RESIDUAL FORCE             3.640E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.111       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        8.113E-04   AT NODE         88   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.713E-09   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.207E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.246E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.209E-04   TIME AVG. FORCE       2.835E-04
 LARGEST RESIDUAL FORCE             1.202E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.111       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.952E-05   AT NODE        108   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.442E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.207E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.497E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.207E-02,  FRACTION OF STEP COMPLETED  0.790    
 STEP TIME COMPLETED        7.90    ,  TOTAL TIME COMPLETED         7.90    


  INCREMENT    65 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.207E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.526E-04   TIME AVG. FORCE       2.892E-04
 LARGEST RESIDUAL FORCE             0.272       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.111       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.770E-04   AT NODE          1   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.617E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.207E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.821E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.260E-04   TIME AVG. FORCE       2.888E-04
 LARGEST RESIDUAL FORCE             7.472E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.111       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.412E-04   AT NODE         89   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.834E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.207E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.271E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.260E-04   TIME AVG. FORCE       2.888E-04
 LARGEST RESIDUAL FORCE             4.993E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.111       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.185E-05   AT NODE        109   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.449E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.207E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.018E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    7.811E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.207E-02,  FRACTION OF STEP COMPLETED  0.795    
 STEP TIME COMPLETED        7.95    ,  TOTAL TIME COMPLETED         7.95    


  INCREMENT    66 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.811E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.046E-04   TIME AVG. FORCE       2.951E-04
 LARGEST RESIDUAL FORCE             0.510       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.167       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.086E-03   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.554E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.811E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.970E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.337E-04   TIME AVG. FORCE       2.940E-04
 LARGEST RESIDUAL FORCE             2.293E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.167       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.258E-04   AT NODE         89   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.097E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.811E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.523E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.337E-04   TIME AVG. FORCE       2.940E-04
 LARGEST RESIDUAL FORCE             4.742E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.167       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.016E-05   AT NODE        108   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.271E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      7.811E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.393E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.117    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.811E-02,  FRACTION OF STEP COMPLETED  0.803    
 STEP TIME COMPLETED        8.03    ,  TOTAL TIME COMPLETED         8.03    


  INCREMENT    67 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.117    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.577E-04   TIME AVG. FORCE       3.024E-04
 LARGEST RESIDUAL FORCE              1.15       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.252       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.457E-03   AT NODE          1   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.701E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.172E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.212E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.453E-04   TIME AVG. FORCE       2.992E-04
 LARGEST RESIDUAL FORCE             6.140E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.252       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.448E-03   AT NODE        129   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.009E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.172E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.027E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.453E-04   TIME AVG. FORCE       2.992E-04
 LARGEST RESIDUAL FORCE             3.604E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.252       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.858E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.435E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.172E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.710E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.176    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.117    ,  FRACTION OF STEP COMPLETED  0.815    
 STEP TIME COMPLETED        8.15    ,  TOTAL TIME COMPLETED         8.15    


  INCREMENT    68 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.176    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 100 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.208E-03   TIME AVG. FORCE       3.126E-04
 LARGEST RESIDUAL FORCE              2.59       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.382       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.280E-02   AT NODE         73   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.044E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.758E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.064E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.637E-04   TIME AVG. FORCE       3.046E-04
 LARGEST RESIDUAL FORCE             3.004E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.382       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.289E-02   AT NODE        216   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.641E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.758E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.865E-10   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.628E-04   TIME AVG. FORCE       3.046E-04
 LARGEST RESIDUAL FORCE             1.620E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.382       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.018E-03   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.889E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.758E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.942E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  3.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.628E-04   TIME AVG. FORCE       3.046E-04
 LARGEST RESIDUAL FORCE             2.620E-10   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.382       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.807E-06   AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.154E-09   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.758E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.304E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.264    

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.176    ,  FRACTION OF STEP COMPLETED  0.832    
 STEP TIME COMPLETED        8.32    ,  TOTAL TIME COMPLETED         8.32    


  INCREMENT    69 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.264    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 287 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.075E-03   TIME AVG. FORCE       3.302E-04
 LARGEST RESIDUAL FORCE              5.80       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.581       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -4.582E-02   AT NODE        810   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.727E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.636E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.830E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 7 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.850E-04   TIME AVG. FORCE       3.115E-04
 LARGEST RESIDUAL FORCE            -0.111       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.581       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.107       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.442E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.636E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.411E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.242E-04   TIME AVG. FORCE       3.107E-04
 LARGEST RESIDUAL FORCE            -5.408E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.581       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.051E-02   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.248E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.636E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.074E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      6.918E-04   TIME AVG. FORCE       3.102E-04
 LARGEST RESIDUAL FORCE            -3.600E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.581       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.559E-02   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.958E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.636E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.245E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      6.903E-04   TIME AVG. FORCE       3.102E-04
 LARGEST RESIDUAL FORCE            -2.033E-04   AT NODE        775   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.581       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.034E-02   AT NODE        775   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.753E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.636E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.117E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      6.896E-04   TIME AVG. FORCE       3.102E-04
 LARGEST RESIDUAL FORCE            -1.089E-06   AT NODE        775   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.581       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.360E-04   AT NODE        775   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.443E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.636E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.560E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.264    ,  FRACTION OF STEP COMPLETED  0.859    
 STEP TIME COMPLETED        8.59    ,  TOTAL TIME COMPLETED         8.59    


  INCREMENT    70 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.264    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 326 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.114E-03   TIME AVG. FORCE       4.216E-04
 LARGEST RESIDUAL FORCE              7.00       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.590       AT NODE        144   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        0.378       AT NODE        865   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.188E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      2.636E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.147E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 153 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT PART-1-1.234 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.276 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.276 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.276 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.276 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.277 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.277 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.285 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.285 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.285 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.287 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.287 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.319 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.320 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.326 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.328 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.328 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.328 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.328 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.336 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.336 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.336 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.336 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.337 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.337 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.337 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.337 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.338 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.338 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.368 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.371 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.371 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.380 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.380 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.381 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.381 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.382 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.383 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.386 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.386 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.386 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.386 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.388 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.388 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.389 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.391 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.391 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.395 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.412 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.421 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.421 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.421 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.431 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.432 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.435 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.436 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.437 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.437 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.439 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.440 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.440 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.447 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.447 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.456 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.471 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.474 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.474 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.475 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.479 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.479 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.487 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.488 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.489 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.490 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.491 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.491 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.491 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.491 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.492 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.496 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.499 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.507 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.523 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.523 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.523 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.549 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.549 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.575 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.575 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.576 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.594 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.599 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.599 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.625 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.625 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.625 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.626 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.626 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.627 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.627 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.633 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.634 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.634 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.642 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.642 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.643 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.645 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.649 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.649 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.650 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.651 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.651 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.651 IS DISTORTING EXCESSIVELY.

 ***WARNING: The anisotropic hyperelastic strain energy potential cannot be 
             evaluated in 1 elements due to unrealistic material properties. 
             These elements have been identified in element set 
             WarnElemAnisoHypelPropStep1Inc70.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 145 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    70 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  6.591E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.927E-04   TIME AVG. FORCE       3.185E-04
 LARGEST RESIDUAL FORCE              1.09       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.147       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.401E-03   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.283E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.591E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.194E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      6.964E-04   TIME AVG. FORCE       3.157E-04
 LARGEST RESIDUAL FORCE             6.273E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.147       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.400E-03   AT NODE         88   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.062E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.591E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.676E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      6.964E-04   TIME AVG. FORCE       3.157E-04
 LARGEST RESIDUAL FORCE             3.576E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.147       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.687E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -8.977E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.591E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.866E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.591E-02,  FRACTION OF STEP COMPLETED  0.865    
 STEP TIME COMPLETED        8.65    ,  TOTAL TIME COMPLETED         8.65    


  INCREMENT    71 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.591E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.559E-04   TIME AVG. FORCE       3.219E-04
 LARGEST RESIDUAL FORCE             0.436       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.147       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -9.676E-04   AT NODE          1   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.864E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.591E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.935E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.032E-04   TIME AVG. FORCE       3.211E-04
 LARGEST RESIDUAL FORCE             1.750E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.147       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.676E-04   AT NODE         89   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.036E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.591E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -3.180E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.032E-04   TIME AVG. FORCE       3.211E-04
 LARGEST RESIDUAL FORCE             2.696E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.147       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.108E-05   AT NODE        109   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.347E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      6.591E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.405E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          TIME INCREMENT MAY NOW INCREASE TO    9.886E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.591E-02,  FRACTION OF STEP COMPLETED  0.872    
 STEP TIME COMPLETED        8.72    ,  TOTAL TIME COMPLETED         8.72    


  INCREMENT    72 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.886E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.476E-04   TIME AVG. FORCE       3.285E-04
 LARGEST RESIDUAL FORCE             0.818       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.222       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.822E-03   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.650E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.886E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.824E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.135E-04   TIME AVG. FORCE       3.266E-04
 LARGEST RESIDUAL FORCE             4.772E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.222       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.047E-03   AT NODE         88   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.899E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.886E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.086E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.135E-04   TIME AVG. FORCE       3.266E-04
 LARGEST RESIDUAL FORCE             2.016E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.222       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        9.980E-05   AT NODE        108   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.920E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      9.886E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.508E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          TIME INCREMENT MAY NOW INCREASE TO    0.148    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.886E-02,  FRACTION OF STEP COMPLETED  0.882    
 STEP TIME COMPLETED        8.82    ,  TOTAL TIME COMPLETED         8.82    


  INCREMENT    73 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.148    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 20 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.219E-03   TIME AVG. FORCE       3.388E-04
 LARGEST RESIDUAL FORCE              1.89       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.335       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.236E-02   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.041E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.728E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.612E-04   TIME AVG. FORCE       3.325E-04
 LARGEST RESIDUAL FORCE             1.781E-02   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.335       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.066E-02   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.836E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.060E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.393E-04   TIME AVG. FORCE       3.322E-04
 LARGEST RESIDUAL FORCE            -5.815E-03   AT NODE        415   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.335       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.744E-02   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.147E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.048E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.292E-04   TIME AVG. FORCE       3.321E-04
 LARGEST RESIDUAL FORCE             8.185E-05   AT NODE        450   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.335       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        6.257E-03   AT NODE        450   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.319E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.430E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      7.291E-04   TIME AVG. FORCE       3.321E-04
 LARGEST RESIDUAL FORCE             8.908E-06   AT NODE        378   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.335       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.559E-03   AT NODE        378   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.951E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.994E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      7.291E-04   TIME AVG. FORCE       3.321E-04
 LARGEST RESIDUAL FORCE             2.803E-10   AT NODE        450   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.335       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.051E-05   AT NODE        487   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.425E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.891E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   6 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  6 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.148    ,  FRACTION OF STEP COMPLETED  0.897    
 STEP TIME COMPLETED        8.97    ,  TOTAL TIME COMPLETED         8.97    


  INCREMENT    74 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.148    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 60 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.902E-03   TIME AVG. FORCE       4.074E-04
 LARGEST RESIDUAL FORCE              3.45       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.380       AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.246       AT NODE        206   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.668E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.032E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 113 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      1.302E-02   TIME AVG. FORCE       5.036E-04
 LARGEST RESIDUAL FORCE             -1.22       AT NODE        287   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.550       AT NODE        252   DOF  3
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -0.556       AT NODE        288   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.898E-02   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.483E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.975E-05   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 303 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.1 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.2 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.2 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.2 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.3 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.3 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.3 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.3 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.3 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.3 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.4 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.4 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.5 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.5 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.6 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.6 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.6 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.6 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.7 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.8 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.9 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.9 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.9 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.9 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.9 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.9 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.10 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.10 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.12 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.12 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.12 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.12 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.12 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.12 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.13 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.13 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.13 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.13 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.14 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.14 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.14 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.14 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.15 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.15 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.15 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.15 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.16 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.17 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.17 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.17 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.17 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.17 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.17 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.18 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.18 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.18 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.19 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.19 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.20 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.20 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.20 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.20 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.20 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.20 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.21 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.22 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.23 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.23 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.23 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.23 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.23 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.23 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.24 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.25 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.25 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.25 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.25 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.26 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.26 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.26 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.26 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.27 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.27 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.27 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.27 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.27 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.27 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.29 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.29 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.29 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.29 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.31 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.31 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.31 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.31 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.31 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.32 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.32 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.32 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.32 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.33 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.33 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.34 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.34 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.34 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.34 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.35 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.35 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.36 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.36 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.36 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.36 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.37 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.37 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.37 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.37 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.38 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.38 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.38 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.38 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.39 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.39 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.39 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.39 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.40 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.40 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.41 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.42 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.42 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.42 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.42 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.42 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.42 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.43 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.43 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.43 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.44 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.44 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.44 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.44 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.45 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.46 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.46 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.46 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.46 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.47 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.47 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.47 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.47 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.47 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.47 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.48 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.48 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.48 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.48 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.48 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.48 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.49 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.49 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.49 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.49 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.50 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.51 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.51 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.51 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.51 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.51 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.52 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.52 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.52 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.53 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.54 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.54 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.54 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.54 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.54 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.54 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.54 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.55 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.55 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.55 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.55 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.55 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.56 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.56 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.56 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.56 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.56 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.56 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.56 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.57 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.57 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.57 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.57 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.58 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.58 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.58 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.59 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.59 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.59 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.59 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.59 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.59 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.59 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.60 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.60 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.60 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.60 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.61 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.61 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.61 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.61 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.62 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.62 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.62 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.62 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.63 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.63 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.63 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.63 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.64 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.64 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.65 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.65 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.65 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.66 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.66 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.66 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.66 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.66 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.66 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.67 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.67 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.67 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.68 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.68 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.68 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.69 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.69 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.69 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.69 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.70 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.70 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.70 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.70 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.70 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.70 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.71 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.71 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.71 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.71 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.72 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.72 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.72 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.72 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.73 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.73 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.73 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.74 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.74 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.75 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.75 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.75 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.75 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.76 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.76 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.77 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.77 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.77 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.78 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.78 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.78 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.78 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.78 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.79 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.79 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.79 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.79 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.80 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.80 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.80 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.80 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.81 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.81 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.81 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.81 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.81 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.82 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.82 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.82 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.83 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.83 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.83 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.83 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.83 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.84 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.85 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.85 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.85 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.85 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.85 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.85 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.85 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.86 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.86 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.86 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.86 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.87 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.88 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.88 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.88 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.89 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.89 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.89 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.89 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.89 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.90 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.91 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.91 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.91 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.91 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.91 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.92 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.92 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.93 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.94 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.94 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.94 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.94 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.94 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.94 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.94 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.95 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.95 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.96 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.96 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.96 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.96 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.97 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.97 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.97 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.97 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.98 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.98 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.99 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.99 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.99 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.99 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.100 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.100 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.100 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.100 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.100 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.100 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.101 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.102 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.103 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.103 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.103 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.103 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.104 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.104 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.104 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.104 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.104 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.104 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.105 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.105 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.105 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.105 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.105 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.106 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.107 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.107 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.107 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.107 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.108 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.108 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.108 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.108 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.108 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.109 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.109 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.109 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.111 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.111 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.111 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.111 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.112 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.112 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.112 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.112 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.112 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.112 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.112 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.113 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.113 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.113 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.113 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.113 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.113 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.113 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.114 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.114 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.114 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.114 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.115 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.115 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.115 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.115 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.115 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.116 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.116 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.116 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.116 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.117 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.118 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.118 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.118 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.118 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.118 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.119 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.120 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.120 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.120 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.120 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.120 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.120 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.121 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.121 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.121 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.121 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.122 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.122 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.122 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.122 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.122 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.122 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.123 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.123 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.123 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.123 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.124 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.124 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.124 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.125 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.125 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.125 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.126 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.126 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.126 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.126 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.126 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.127 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.127 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.128 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.128 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.128 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.129 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.129 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.130 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.130 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.130 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.131 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.131 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.131 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.132 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.132 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.132 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.133 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.133 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.133 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.133 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.133 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.134 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.134 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.134 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.134 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.135 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.135 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.135 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.136 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.136 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.136 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.136 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.137 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.137 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.137 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.137 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.137 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.138 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.138 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.138 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.138 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.138 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.138 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.139 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.139 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.140 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.140 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.141 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.141 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.141 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.141 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.141 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.141 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.142 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.142 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.142 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.142 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.142 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.143 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.143 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.143 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.144 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.145 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.145 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.145 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.145 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.145 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.145 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.146 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.146 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.146 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.146 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.146 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.147 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.147 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.148 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.148 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.148 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.149 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.149 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.149 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.150 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.150 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.150 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.150 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.151 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.151 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.151 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.152 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.152 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.152 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.152 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.152 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.153 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.154 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.154 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.154 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.154 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.155 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.155 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.155 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.155 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.155 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.157 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.157 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.157 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.157 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.157 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.157 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.158 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.158 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.158 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.158 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.158 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.159 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.159 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.159 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.160 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.160 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.160 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.160 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.161 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.161 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.161 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.161 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.161 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.162 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.162 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.163 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.163 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.163 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.164 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.164 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.164 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.164 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.164 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.165 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.166 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.166 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.166 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.166 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.166 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.167 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.167 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.167 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.168 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.168 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.168 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.169 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.170 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.170 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.170 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.170 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.170 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.171 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.172 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.172 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.173 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.173 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.173 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.173 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.173 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.173 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.173 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.174 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.174 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.174 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.174 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.174 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.175 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.175 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.175 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.176 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.176 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.176 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.176 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.176 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.176 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.176 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.177 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.177 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.177 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.178 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.178 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.178 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.178 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.179 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.179 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.179 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.180 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.180 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.180 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.180 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.180 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.180 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.181 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.181 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.181 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.182 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.182 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.182 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.182 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.182 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.182 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.183 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.183 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.183 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.184 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.184 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.184 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.184 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.184 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.185 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.185 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.185 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.185 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.186 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.186 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.186 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.186 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.186 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.187 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.188 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.188 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.188 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.189 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.190 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.190 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.190 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.190 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.191 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.191 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.191 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.192 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.192 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.192 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.192 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.193 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.193 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.193 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.193 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.193 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.193 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.194 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.194 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.194 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.194 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.194 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.194 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.195 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.195 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.195 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.195 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.195 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.196 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.196 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.196 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.196 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.197 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.197 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.197 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.198 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.198 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.198 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.198 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.198 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.198 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.199 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.199 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.199 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.199 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.199 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.199 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.201 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.201 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.201 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.201 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.201 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.201 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.201 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.202 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.202 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.202 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.202 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.203 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.204 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.205 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.206 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.206 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.206 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.206 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.206 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.206 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.207 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.207 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.208 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.208 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.208 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.208 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.208 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.209 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.209 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.209 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.209 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.209 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.210 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.210 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.211 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.211 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.211 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.211 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.211 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.211 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.212 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.212 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.212 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.212 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.213 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.213 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.213 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.214 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.214 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.214 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.214 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.215 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.215 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.215 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.216 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.216 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.216 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.216 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.216 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.216 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.217 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.217 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.217 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.217 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.217 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.217 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.218 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.219 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.219 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.219 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.219 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.219 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.220 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.220 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.220 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.220 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.221 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.221 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.222 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.222 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.222 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.222 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.223 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.223 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.224 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.224 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.224 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.224 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.224 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.224 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.225 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.225 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.225 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.225 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.226 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.226 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.226 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.226 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.226 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.227 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.227 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.227 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.227 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.227 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.228 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.228 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.228 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.229 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.229 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.229 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.230 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.230 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.230 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.230 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.230 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.231 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.231 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.231 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.231 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.231 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.231 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.232 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.232 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.232 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.232 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.233 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.233 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.233 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.234 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.234 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.234 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.234 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.234 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.235 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.235 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.235 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.236 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.236 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.236 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.237 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.237 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.237 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.238 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.239 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.240 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.241 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.241 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.241 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.241 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.241 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.241 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.242 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.242 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.242 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.242 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.242 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.242 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.243 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.243 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.243 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.243 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.243 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.243 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.243 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.244 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.244 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.245 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.245 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.245 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.246 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.246 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.246 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.246 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.246 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.246 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.247 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.247 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.247 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.248 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.248 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.248 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.248 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.248 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.248 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.248 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.249 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.249 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.249 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.250 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.250 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.250 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.250 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.250 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.250 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.251 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.251 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.251 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.252 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.252 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.252 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.253 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.253 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.253 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.253 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.253 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.255 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.256 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.257 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.257 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.257 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.258 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.258 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.259 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.259 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.259 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.259 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.259 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.260 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.260 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.260 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.260 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.261 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.261 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.261 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.261 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.261 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.262 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.262 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.262 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.263 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.263 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.264 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.264 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.265 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.265 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.265 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.265 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.266 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.267 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.267 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.267 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.267 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.267 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.268 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.269 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.269 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.269 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.269 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.269 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.270 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.270 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.270 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.271 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.271 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.271 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.271 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.272 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.272 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.272 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.272 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.273 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.273 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.273 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.273 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.273 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.273 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.273 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.274 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.274 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.274 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.274 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.274 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.274 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.275 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.276 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.276 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.276 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.277 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.277 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.277 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.277 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.278 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.278 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.278 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.278 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.279 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.279 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.280 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.280 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.280 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.280 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.280 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.281 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.282 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.282 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.282 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.282 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.282 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.283 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.283 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.283 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.283 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.284 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.284 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.284 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.284 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.285 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.285 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.286 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.287 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.288 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.288 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.288 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.288 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.288 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.288 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.289 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.289 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.289 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.289 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.289 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.290 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.290 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.290 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.290 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.291 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.291 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.292 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.292 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.292 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.292 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.292 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.293 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.293 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.293 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.293 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.293 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.293 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.294 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.294 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.294 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.294 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.295 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.295 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.296 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.297 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.298 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.298 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.298 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.299 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.299 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.299 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.299 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.299 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.299 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.300 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.300 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.300 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.300 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.300 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.301 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.301 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.301 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.302 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.302 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.302 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.302 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.303 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.303 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.303 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.303 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.303 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.304 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.304 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.304 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.304 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.304 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.304 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.305 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.305 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.305 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.306 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.307 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.307 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.307 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.307 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.307 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.308 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.308 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.308 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.309 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.309 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.309 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.309 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.310 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.311 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.311 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.311 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.311 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.311 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.311 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.311 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.312 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.312 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.312 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.313 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.313 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.313 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.314 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.314 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.314 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.315 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.315 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.315 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.315 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.316 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.316 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.316 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.316 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.317 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.318 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.319 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.319 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.319 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.320 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.320 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.320 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.321 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.321 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.321 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.322 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.322 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.322 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.322 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.322 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.323 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.323 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.323 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.324 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.324 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.324 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.324 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.325 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.325 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.326 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.326 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.326 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.326 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.327 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.328 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.329 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.329 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.329 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.329 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.329 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.329 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.329 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.330 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.330 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.330 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.330 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.330 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.331 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.331 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.331 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.331 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.332 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.333 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.334 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.334 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.334 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.334 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.335 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.335 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.335 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.335 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.335 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.336 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.336 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.336 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.338 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.338 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.339 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.339 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.339 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.339 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.339 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.339 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.339 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.340 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.340 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.340 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.340 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.341 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.342 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.342 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.342 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.342 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.342 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.342 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.343 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.343 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.343 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.343 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.344 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.344 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.344 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.346 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.347 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.348 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.348 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.348 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.348 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.349 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.349 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.349 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.349 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.350 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.351 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.351 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.351 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.351 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.352 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.352 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.352 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.352 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.352 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.352 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.352 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.353 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.353 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.353 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.353 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.355 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.355 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.355 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.355 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.355 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.356 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.356 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.356 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.356 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.357 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.357 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.357 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.357 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.357 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.358 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.359 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.360 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.360 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.360 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.360 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.361 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.361 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.361 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.361 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.362 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.362 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.364 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.364 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.364 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.364 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.364 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.365 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.365 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.365 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.366 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.366 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.366 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.367 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.367 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.367 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.367 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.367 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.368 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.368 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.368 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.369 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.369 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.369 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.369 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.369 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.371 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.371 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.372 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.372 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.372 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.372 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.372 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.372 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.373 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.373 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.373 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.373 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.373 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.373 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.373 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.374 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.375 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.376 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.376 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.376 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.376 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.377 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.377 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.377 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.377 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.377 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.377 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.378 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.378 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.378 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.378 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.378 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.378 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.379 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.379 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.379 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.380 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.381 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.381 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.382 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.384 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.385 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.386 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.387 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.387 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.388 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.389 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.389 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.389 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.389 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.389 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.390 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.391 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.391 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.391 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.391 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.392 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.393 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.393 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.393 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.393 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.393 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.394 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.394 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.394 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.394 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.394 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.395 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.395 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.395 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.395 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.397 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.397 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.398 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.398 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.398 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.398 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.398 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.399 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.399 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.399 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.400 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.401 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.401 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.401 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.402 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.403 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.403 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.403 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.403 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.403 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.403 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.404 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.404 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.404 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.404 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.404 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.404 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.405 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.407 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.407 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.407 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.407 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.408 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.408 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.408 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.408 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.409 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.410 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.410 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.411 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.411 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.411 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.411 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.412 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.412 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.412 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.412 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.415 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.415 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.416 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.416 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.416 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.417 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.417 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.417 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.417 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.418 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.418 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.418 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.418 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.418 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.418 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.419 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.420 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.421 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.421 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.421 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.422 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.422 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.422 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.422 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.423 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.423 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.423 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.423 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.424 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.424 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.424 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.424 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.425 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.425 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.425 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.425 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.425 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.426 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.426 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.427 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.428 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.428 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.428 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.428 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.428 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.428 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.428 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.429 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.430 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.430 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.430 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.430 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.433 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.434 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.437 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.437 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.439 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.440 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.440 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.440 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.441 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.441 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.441 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.442 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.442 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.442 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.442 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.442 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.443 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.444 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.445 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.445 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.445 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.446 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.448 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.449 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.450 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.450 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.453 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.453 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.454 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.454 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.454 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.454 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.454 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.454 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.455 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.455 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.455 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.457 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.457 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.458 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.458 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.458 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.458 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.458 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.458 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.459 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.460 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.461 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.461 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.461 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.461 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.462 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.462 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.463 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.466 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.467 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.468 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.468 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.468 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.469 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.469 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.469 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.469 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.470 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.470 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.470 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.471 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.471 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.471 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.472 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.473 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.473 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.473 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.474 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.474 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.474 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.474 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.475 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.475 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.475 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.475 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.476 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.477 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.478 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.478 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.478 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.479 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.479 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.479 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.480 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.480 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.485 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.487 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.490 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.491 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.491 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.491 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.492 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.492 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.492 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.493 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.494 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.495 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.496 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.497 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.500 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.500 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.501 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.501 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.501 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.502 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.505 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.508 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.508 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.508 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.508 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.509 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.509 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.509 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.509 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.510 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.511 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.512 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.512 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.513 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.513 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.513 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.513 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.514 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.516 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.517 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.518 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.518 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.519 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.519 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.520 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.520 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.521 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.521 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.522 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.522 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.522 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.524 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.524 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.524 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.524 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.525 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.525 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.526 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.526 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.526 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.526 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.527 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.528 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.529 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.529 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.529 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.529 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.535 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.537 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.541 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.543 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.545 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.546 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.546 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.547 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.548 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.550 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.551 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.551 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.551 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.552 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.553 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.554 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.559 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.560 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.562 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.562 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.562 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.563 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.563 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.563 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.563 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.563 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.563 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.564 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.565 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.565 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.565 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.566 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.566 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.567 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.567 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.568 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.568 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.569 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.569 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.570 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.570 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.571 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.571 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.572 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.573 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.574 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.574 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.575 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.576 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.576 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.576 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.577 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.577 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.578 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.579 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.579 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.579 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.579 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.580 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.580 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.580 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.581 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.583 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.585 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.587 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.593 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.597 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.597 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.597 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.598 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.599 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.600 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.601 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.602 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.603 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.603 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.604 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.613 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.614 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.614 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.614 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.614 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.615 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.616 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.617 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.617 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.617 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.617 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.618 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.618 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.618 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.619 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.619 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.620 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.620 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.621 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.621 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.622 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.622 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.623 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.624 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.626 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.626 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.627 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.628 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.628 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.628 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.630 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.630 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.630 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.630 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.631 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.631 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.632 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.632 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.632 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.633 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.635 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.637 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.647 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.649 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.650 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.651 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.651 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.652 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.653 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.653 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.654 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.655 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.660 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.661 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.664 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.665 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.666 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.667 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.668 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.669 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.669 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.669 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.670 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.670 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.671 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.671 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.672 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.672 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.673 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.674 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.674 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.678 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.680 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.681 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.681 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.682 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.682 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.682 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.682 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.682 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.683 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.683 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.683 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.683 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.684 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.684 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.685 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.687 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.689 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.701 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.703 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.703 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.704 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.705 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.710 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.710 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.710 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.710 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.711 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.712 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.713 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.713 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.713 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.713 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.714 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.714 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.715 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.716 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.717 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.718 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.719 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.720 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.720 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.720 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.721 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.721 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.722 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.722 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.723 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.724 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.729 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.729 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.730 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.731 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.732 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.732 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.733 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.733 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.733 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.733 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.733 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.734 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.734 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.734 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.734 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.735 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.737 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.739 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.744 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.744 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.744 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.745 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.745 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.745 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.745 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.746 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.746 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.746 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.746 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.746 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.746 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.747 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.747 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.747 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.747 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.747 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.748 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.748 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.748 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.748 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.753 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.755 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.757 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.760 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.760 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.760 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.760 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.760 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.760 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.760 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.761 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.762 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.763 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.764 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.765 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.766 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.767 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.768 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.768 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.768 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.768 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.769 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.769 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.769 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.769 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.770 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.770 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.772 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.774 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.774 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.774 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.775 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.775 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.775 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.776 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.776 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.777 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.777 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.777 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.779 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.779 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.779 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.779 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.780 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.780 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.780 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.781 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.781 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.781 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.781 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.782 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.782 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.782 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.782 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.783 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.784 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.784 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.784 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.785 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.793 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.793 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.794 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.794 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.794 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.794 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.794 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.794 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.795 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.795 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.795 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.795 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.795 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.795 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.796 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.797 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.798 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.799 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.807 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.808 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.808 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.809 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.809 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.809 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.809 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.809 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.809 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.809 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.810 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.811 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.812 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.813 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.814 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.815 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.816 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.817 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.817 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.817 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.817 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.818 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.818 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.818 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.818 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.820 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.820 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.820 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.820 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.822 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.823 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.823 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.823 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.823 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.824 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.824 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.825 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.825 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.825 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.825 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.826 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.826 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.827 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.828 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.828 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.829 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.829 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.830 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.832 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.834 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.835 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.836 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.837 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.843 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.843 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.843 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.843 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.845 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.845 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.846 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.846 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.846 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.847 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.847 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.847 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.847 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.848 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.848 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.848 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.849 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.849 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.849 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.849 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.849 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.849 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.849 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.850 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.850 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.850 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.852 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.852 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.858 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.858 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.860 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.860 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.860 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.860 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.860 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.860 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.860 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.861 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.862 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.863 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.864 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.865 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.866 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.
   ELEMENT PART-1-1.867 IS DISTORTING EXCESSIVELY.

 ***WARNING: The anisotropic hyperelastic strain energy potential cannot be 
             evaluated in 1 elements due to unrealistic material properties. 
             These elements have been identified in element set 
             WarnElemAnisoHypelPropStep1Inc74.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 2632 INTEGRATION POINTS IN 
             SOLID (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT    74 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  3.707E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.685E-04   TIME AVG. FORCE       3.380E-04
 LARGEST RESIDUAL FORCE             0.345       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.405E-02   AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.768E-04   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.543E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.707E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.881E-09   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.329E-04   TIME AVG. FORCE       3.375E-04
 LARGEST RESIDUAL FORCE             1.151E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.405E-02   AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.618E-04   AT NODE         89   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.065E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.707E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.392E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.329E-04   TIME AVG. FORCE       3.375E-04
 LARGEST RESIDUAL FORCE             1.164E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.405E-02   AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.933E-05   AT NODE        109   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            4.136E-08   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.707E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     4.305E-08   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.707E-02,  FRACTION OF STEP COMPLETED  0.900    
 STEP TIME COMPLETED        9.00    ,  TOTAL TIME COMPLETED         9.00    


  INCREMENT    75 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.707E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.447E-04   TIME AVG. FORCE       3.429E-04
 LARGEST RESIDUAL FORCE             0.138       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         8.421E-02   AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.120E-04   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.029E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.707E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.040E-09   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.368E-04   TIME AVG. FORCE       3.428E-04
 LARGEST RESIDUAL FORCE             2.043E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -8.422E-02   AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.043E-04   AT NODE         89   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.307E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.707E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.595E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.368E-04   TIME AVG. FORCE       3.428E-04
 LARGEST RESIDUAL FORCE            -2.327E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -8.421E-02   AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.621E-05   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.047E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      3.707E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.458E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          TIME INCREMENT MAY NOW INCREASE TO    5.561E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.707E-02,  FRACTION OF STEP COMPLETED  0.904    
 STEP TIME COMPLETED        9.04    ,  TOTAL TIME COMPLETED         9.04    


  INCREMENT    76 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.561E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.649E-04   TIME AVG. FORCE       3.484E-04
 LARGEST RESIDUAL FORCE             0.259       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.127       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.868E-04   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.215E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.561E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.083E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.427E-04   TIME AVG. FORCE       3.481E-04
 LARGEST RESIDUAL FORCE             6.769E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.127       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.574E-04   AT NODE         89   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.100E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.561E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.052E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.427E-04   TIME AVG. FORCE       3.481E-04
 LARGEST RESIDUAL FORCE             4.007E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.127       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.124E-05   AT NODE        109   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.203E-08   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      5.561E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.324E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    8.341E-02

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.561E-02,  FRACTION OF STEP COMPLETED  0.910    
 STEP TIME COMPLETED        9.10    ,  TOTAL TIME COMPLETED         9.10    


  INCREMENT    77 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.341E-02

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.324E-04   TIME AVG. FORCE       3.544E-04
 LARGEST RESIDUAL FORCE             0.583       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.191       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.323E-03   AT NODE          1   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.522E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.341E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.816E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.516E-04   TIME AVG. FORCE       3.533E-04
 LARGEST RESIDUAL FORCE             2.865E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.191       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -7.730E-04   AT NODE        128   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.947E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.341E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -7.529E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.516E-04   TIME AVG. FORCE       3.533E-04
 LARGEST RESIDUAL FORCE             7.199E-08   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.191       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.251E-05   AT NODE        109   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.493E-07   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      8.341E-03   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.300E-07   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          TIME INCREMENT MAY NOW INCREASE TO    0.125    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.341E-02,  FRACTION OF STEP COMPLETED  0.918    
 STEP TIME COMPLETED        9.18    ,  TOTAL TIME COMPLETED         9.18    


  INCREMENT    78 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.125    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.012E-03   TIME AVG. FORCE       3.618E-04
 LARGEST RESIDUAL FORCE              1.31       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.287       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.989E-03   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.652E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.251E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.941E-08   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      7.651E-04   TIME AVG. FORCE       3.586E-04
 LARGEST RESIDUAL FORCE             5.977E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.287       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.821E-03   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.391E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.251E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -4.257E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.651E-04   TIME AVG. FORCE       3.586E-04
 LARGEST RESIDUAL FORCE             4.233E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.287       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.437E-04   AT NODE        126   DOF  1
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -3.910E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.251E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     6.585E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          TIME INCREMENT MAY NOW INCREASE TO    0.188    

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.125    ,  FRACTION OF STEP COMPLETED  0.930    
 STEP TIME COMPLETED        9.30    ,  TOTAL TIME COMPLETED         9.30    


  INCREMENT    79 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.188    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 123 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.454E-03   TIME AVG. FORCE       3.725E-04
 LARGEST RESIDUAL FORCE              2.97       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.434       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.658E-02   AT NODE        505   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -7.430E-03   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.960E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.017E-04   TIME AVG. FORCE       3.642E-04
 LARGEST RESIDUAL FORCE             8.433E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.434       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.968E-02   AT NODE        505   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.628E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.266E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      7.853E-04   TIME AVG. FORCE       3.640E-04
 LARGEST RESIDUAL FORCE             1.374E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.434       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.291E-03   AT NODE        127   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.753E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.213E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      7.854E-04   TIME AVG. FORCE       3.640E-04
 LARGEST RESIDUAL FORCE            -3.238E-06   AT NODE        523   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.434       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.119E-04   AT NODE        505   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -5.105E-06   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.478E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      7.854E-04   TIME AVG. FORCE       3.640E-04
 LARGEST RESIDUAL FORCE            -4.322E-09   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.434       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -8.389E-06   AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.018E-07   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -6.078E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

 ITERATION SUMMARY FOR THE INCREMENT:   5 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  5 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.188    ,  FRACTION OF STEP COMPLETED  0.949    
 STEP TIME COMPLETED        9.49    ,  TOTAL TIME COMPLETED         9.49    


  INCREMENT    80 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.188    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  6.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 164 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.713E-03   TIME AVG. FORCE       3.809E-04
 LARGEST RESIDUAL FORCE              3.55       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.438       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.794E-02   AT NODE        518   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.992E-03   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.185E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.819E-04   TIME AVG. FORCE       3.705E-04
 LARGEST RESIDUAL FORCE            -7.233E-03   AT NODE        630   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.438       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.328E-02   AT NODE        432   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.768E-03   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     9.086E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.229E-04   TIME AVG. FORCE       3.698E-04
 LARGEST RESIDUAL FORCE            -4.289E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.440       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        5.485E-02   AT NODE        594   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.424E-02   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.591E-04   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.125E-04   TIME AVG. FORCE       3.696E-04
 LARGEST RESIDUAL FORCE             1.500E-03   AT NODE        558   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.439       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.972E-02   AT NODE        522   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            5.179E-03   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -9.810E-05   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.067E-04   TIME AVG. FORCE       3.696E-04
 LARGEST RESIDUAL FORCE             2.227E-04   AT NODE        594   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.        -0.438       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.088E-02   AT NODE        594   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            9.962E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.493E-05   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      8.061E-04   TIME AVG. FORCE       3.696E-04
 LARGEST RESIDUAL FORCE            -2.058E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.438       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.616E-03   AT NODE        522   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            8.916E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.358E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      8.061E-04   TIME AVG. FORCE       3.696E-04
 LARGEST RESIDUAL FORCE            -6.527E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.438       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -5.439E-04   AT NODE         18   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.573E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.465E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     8


 AVERAGE FORCE                      8.061E-04   TIME AVG. FORCE       3.696E-04
 LARGEST RESIDUAL FORCE            -7.532E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.438       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.524E-04   AT NODE          1   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.044E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.024E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

 ITERATION SUMMARY FOR THE INCREMENT:   8 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  8 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.188    ,  FRACTION OF STEP COMPLETED  0.968    
 STEP TIME COMPLETED        9.68    ,  TOTAL TIME COMPLETED         9.68    


  INCREMENT    81 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.188    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 163 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.626E-03   TIME AVG. FORCE       3.851E-04
 LARGEST RESIDUAL FORCE              3.54       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.441       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.112E-02   AT NODE         88   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -4.575E-02   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.912E-07   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.273E-04   TIME AVG. FORCE       3.752E-04
 LARGEST RESIDUAL FORCE             4.573E-04   AT NODE        108   DOF  1
   INSTANCE: PART-1-1                                                                        
 LARGEST INCREMENT OF DISP.         0.442       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.078E-02   AT NODE         88   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.478E-04   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.264E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.272E-04   TIME AVG. FORCE       3.752E-04
 LARGEST RESIDUAL FORCE            -1.768E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.442       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        3.611E-03   AT NODE         90   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -2.380E-04   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -1.401E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.271E-04   TIME AVG. FORCE       3.752E-04
 LARGEST RESIDUAL FORCE            -5.514E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.442       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -2.843E-04   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            1.025E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.877E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.646E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

 ITERATION SUMMARY FOR THE INCREMENT:   4 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  4 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.188    ,  FRACTION OF STEP COMPLETED  0.987    
 STEP TIME COMPLETED        9.87    ,  TOTAL TIME COMPLETED         9.87    


  INCREMENT    82 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.133    

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 45 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.306E-03   TIME AVG. FORCE       3.866E-04
 LARGEST RESIDUAL FORCE              2.16       AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.315       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -1.564E-02   AT NODE        153   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            3.942E-02   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.059E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =  1.00000E-02
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      8.471E-04   TIME AVG. FORCE       3.810E-04
 LARGEST RESIDUAL FORCE             6.832E-03   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.315       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        1.608E-02   AT NODE        153   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            6.690E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     3.047E-06   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  3.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  3.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      8.421E-04   TIME AVG. FORCE       3.809E-04
 LARGEST RESIDUAL FORCE             7.329E-04   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.316       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        4.107E-03   AT NODE         91   DOF  1
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -1.928E-03   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     8.897E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     4


 AVERAGE FORCE                      8.421E-04   TIME AVG. FORCE       3.809E-04
 LARGEST RESIDUAL FORCE            -5.843E-05   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.315       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        2.707E-03   AT NODE          1   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -9.120E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -8.615E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     5


 AVERAGE FORCE                      8.421E-04   TIME AVG. FORCE       3.809E-04
 LARGEST RESIDUAL FORCE            -6.685E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.315       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -3.345E-04   AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.614E-05   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -2.444E-05   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  5.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  5.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     6


 AVERAGE FORCE                      8.421E-04   TIME AVG. FORCE       3.809E-04
 LARGEST RESIDUAL FORCE            -3.171E-06   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.        -0.315       AT NODE          1   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.       -6.266E-04   AT NODE         72   DOF  3
   INSTANCE: PART-1-1                                                                        
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT            2.182E-05   AT NODE          1   DOF  6
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.034E-05   AT NODE          1   DOF  5
   INSTANCE: {assembly node}                                                                 
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO

  NUMBER OF EQUATIONS =      3678     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  1.39E+08

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  4.00000E-02
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  4.00000E-02
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     7


 AVERAGE FORCE                      8.421E-04   TIME AVG. FORCE       3.809E-04
 LARGEST RESIDUAL FORCE            -3.764E-07   AT NODE          1   DOF  1
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF DISP.         0.315       AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
 LARGEST CORRECTION TO DISP.        7.916E-05   AT NODE         72   DOF  2
   INSTANCE: PART-1-1                                                                        
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST RESIDUAL MOMENT           -6.207E-06   AT NODE          1   DOF  5
   INSTANCE:                                                                                 
 LARGEST INCREMENT OF ROTATION      1.331E-02   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.789E-06   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          ALL ROTATION CORRECTIONS ARE ZERO

 ITERATION SUMMARY FOR THE INCREMENT:   7 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  7 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.133    ,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        10.0    ,  TOTAL TIME COMPLETED         10.0    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF         82  INCREMENTS
                      13  CUTBACKS IN AUTOMATIC INCREMENTATION
                     376  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     376  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                     376  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       1  WARNING MESSAGES DURING USER INPUT PROCESSING
                     191  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                     182  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   75.420    
       SYSTEM TIME (SEC)    =   4.4600    
       TOTAL CPU TIME (SEC) =   79.880    
       WALLCLOCK TIME (SEC) =         84
