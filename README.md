
<div style="text-align:center;">
<h1>Frost Bite</h1>
<img src="./images/frostbite.png" width="200" />
</div>

# Ojectifs

# Environnement

Todo Open-ai gym.

Gnuplot

# Agents

## Sarsa

```console
+-------------------------------------+
+ Agent: Sarsa                        +
+-------------------------------------+
+ epsilon: 0.2
+ alpha: 0.5
+ gamma: 0.99
+-------------------------------------+
+ Episode 1000              score: 150.0
+ Mean of last 50 = 74.2   Highest Score: 230.0
+-------------------------------------+
  250 +-----------------------------------------------------------------------------------------+
      |                                                                                         |
      |                *                                                                        |
      |                *         *                                                              |
      |                **        *                                                              |
  200 |                **        *                                                              |
      |                **        *        *           *               *                         |
      |    *           **       **    *  **     *    **  *   *      * * *                       |
      |   **   *    *  **       **    *  **     *    **  **  *      * * * *       *             |
      |   **   *   **  ****     ** *  ** ***    *    **  **  *   *  * * * *       *  *         *|
  150 |   **** * * *** ****     ** * *** **** * *  * ** *** ** * *  * * * **      *  *    *  * *|
      |* ***** * * ********  *  **** *** **** * *  *********** * *  * ****** *  * *  *    ** * *|
      |* ********* ********  ** **** ********** *  *********** ***  * ****** *  * ** **   ** ***|
      |* ********* ********  ** **** ********** *  *********** ***  * ****** *  * ** **   ** ***|
      |* ********* ******** *** **** ************  *********** ****** ****** *  **** **  *** ***|
      |******************************************* *********** ************* ******* *** *** ***|
  100 |******************************************************* ************* ******* ***********|
      |*****************************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
   50 |*************************************************************************** *************|
      |**************************************************************** *** ****** ****** ******|
      |********* ****** *********************** ****************** **** *** ****** ****** ** ***|
      |****** *  *****  * *****   * ***  * **** * * ******** ***** * *  *** ** *** ****** ** ***|
      |** **     * ***  *     *   *  *   * ***  * * **** **   *  * * *      *  *** ** * * **  **|
    0 +-----------------------------------------------------------------------------------------+
      0       100      200      300      400      500      600      700      800      900      100
```

```
+-------------------------------------+
+ Agent: Sarsa                        +
+-------------------------------------+
+ epsilon: 0.4
+ alpha: 0.1
+ gamma: 0.99
+-------------------------------------+
+ Episode 1000              score: 80.0
+ Mean of last 50 = 82.4   Highest Score: 260.0
+-------------------------------------+
  300 +-----------------------------------------------------------------------------------------+
      |                                                                                         |
      |                                                                                         |
      |     *                                                                                   |
  250 |     *                                                                                   |
      |     *                                                                                   |
      |     *                                                                                   |
      |     * *                                                                                 |
      |     * *                                                                                 |
  200 |     * **                                                         *                      |
      |     * **        *    *     * *        *  *         *             *            *   *  *  |
      |   * * **       **    *     * *        *  *       * *      *    * *   *  **  * *   * *** |
      |   * * ** *  *  **    *    ** **  *   **  *      ** *     **   ** *   *  **  * *   * *** |
  150 |   * *********  **   **    ** *** *   *** *** * ***** **  ** * *****  *  ** ** **  * ****|
      |   ****************  **  * ****** *   ******* ******* **  *********** *  ***** **  * ****|
      | * ****************  ** ** ********   ******* ********** ************** *********  * ****|
      |******************* *********************************************************************|
  100 |*****************************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
   50 |*****************************************************************************************|
      |********************************* ******  ********************** ************************|
      |* ****** ** *** ******** * * * *  ******  *    ***** **  ** * *  * **   ****** *  ** *  *|
      |* *** *     * *  ** ***    *   *   ** *   *    * * * **  ** * *  *  *    *****    *  *   |
    0 +-----------------------------------------------------------------------------------------+
      0       100      200      300      400      500      600      700      800      900      1000
    ```

## Reinforce

```console
+-------------------------------------+
+ Agent: Reinforce                    +
+-------------------------------------+
+ learning_rate: 0.001
+ gamma: 0.99
+ hidden1: 36
+ hidden2: 36
+-------------------------------------+
+ Episode 1000              score: 50.0
+ Mean of last 50 = 66.4   Highest Score: 220.0
+-------------------------------------+
  250 +-----------------------------------------------------------------------------------------+
      |                                                                                         |
      |                                                                                         |
      |                                        *                                       *   *    |
      |                                        *                                       *   *    |
  200 |                       *                *   *                                   *   *    |
      |                   *   *                *   *                 *                **   *    |
      |                   * * *        *  *    *   * *               *            *   **  **    |
      |        *          * * *        *  *   **   * *  *          * *            *   **  **    |
      |  * *   *          *** *        *  *   **   * *  *       *  * *       **   *   **  **    |
  150 |  * *   ***  *    **** * *  * * *  *   ***  * * **     * *  * * *    ***   **  **  **  **|
      |  * *   *** ** *  ********* * * ** *** ***  *** **     * *  * * **  ****   *** **  **  **|
      |* * * * ******** ********** * ************* *** *** * ** ** * ***** ****  **** ****** ***|
      |* * * * ******** ********** * ************* *** *** * ** ** * ***** ****  **** ****** ***|
      |*** * ********** ************************** *** ***** ** *************** ************ ***|
      |***** ********** ******************************************************* ****************|
  100 |***** ****************************************************************** ****************|
      |***** ***********************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
   50 |********************* ********************************************************** ********|
      |***** *************** ********************************************************** ********|
      |*****  ************** * ****** ************************************* ******* *** ********|
      | ***   **** ********* * ****** ** **  *** *** **** * ******** *** **  ****** * * *** **  |
      | * *   ***     ****       ** *  * **  *** *   * ** *  *  **    ** **   ***** * * *   *   |
    0 +-----------------------------------------------------------------------------------------+
      0       100      200      300      400      500      600      700      800      900      1000
```

```console
+-------------------------------------+
+ Agent: Reinforce                    +
+-------------------------------------+
+ learning_rate: 0.01
+ gamma: 0.9
+ hidden1: 128
+ hidden2: 128
+ hidden3: 128
+-------------------------------------+
+ Episode 1000              score: 40.0
+ Mean of last 50 = 79.0   Highest Score: 220.0
+-------------------------------------+
  250 +-----------------------------------------------------------------------------------------+
      |                                                                                         |
      |                                                                                         |
      |     *                                                                                   |
      |     *                 *                                                                 |
  200 |     *                 *                                                                 |
      |     *   *           * *         *                                    *                  |
      |  * **   *         * * *         *                                    *                  |
      |  * **   *    *    * * *         *            *       * *             *                **|
      | ** **   *    *   ** * *   *     *      *  ** *      ** *          *  ** * * **  *    ***|
  150 | ** **   *    * * ** * *  ** *   *  *   * *** *   *  ** *    *     *  ** * **** **    ***|
      | ** **   *   ** * **** * *** *  *** *   ***** * * ** ** *    *    ** *** * **** **   ****|
      | ** **   * ****** **** ***** *  *** *   ******* * ***** *    *  **** *** * *******  *****|
      |*** ***  * ****** **** ***** *  *** *   ******* * ***** *    *  **** *** * *******  *****|
      |*** ***  ******** **** ***** ********** ******* * *******    * ***** *** ********* ******|
      |*** *** ************** ************************ * ********* ** **************************|
  100 |*** ****************** ************************ * ********* ** **************************|
      |*** ******************************************* * ***************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
      |*****************************************************************************************|
   50 |*****************************************************************************************|
      |********************************************************************** ******************|
      |********************************************** **** ****************** ******************|
      |** * * * ********** ***** *********** ****  ** * ** **** ** ***** *    ** ** **** *****  |
      |** * * * * *******  *****  ** ***** * ***   *  * *  **   *  ** **      *   *  * *  * **  |
    0 +-----------------------------------------------------------------------------------------+
      0       100      200      300      400      500      600      700      800      900      1000
```

```
+-------------------------------------+
+ Agent: Reinforce                    +
+-------------------------------------+
+ learning_rate: 0.001
+ gamma: 0.5
+ hidden1: 128
+ hidden2: 128
+ hidden2: 18
+-------------------------------------+
+ Episode 7095              score: 30.0
+ Mean of last 50 = 70.8   Highest Score: 260.0
+-------------------------------------+
  300 +-----------------------------------------------------------------------------------------+
      |                                                                                         |
      |                                                                                         |
      |                                  *                                                      |
  250 |                                  *                                                      |
      |               *                  *                                                      |
      |        *      *                  *                                                      |
      |        *      *  *               *         *          *                    *            |
      |   *    *      *  *       *       * *       *       *  ** *                 *            |
  200 |  **  ***     **  *   *   **   *  ***  *   **      **  ** *           *   * *            |
      |* *** *********** *********** **  **** *** ***   **** *** **** * *** ******** *          |
      |********************************  **** ********  **************************** **         |
      |********************************  ******************************************* **         |
  150 |********************************************************************************         |
      |********************************************************************************         |
      |********************************************************************************         |
      |********************************************************************************         |
  100 |********************************************************************************         |
      |********************************************************************************         |
      |********************************************************************************         |
      |********************************************************************************         |
      |********************************************************************************         |
   50 |********************************************************************************         |
      |********************************************************************************         |
      |********************************************************************************         |
      |******** *********************************************** ***********************         |
    0 +-----------------------------------------------------------------------------------------+
      0         1000        2000       3000       4000       5000        6000       7000       8000

```

