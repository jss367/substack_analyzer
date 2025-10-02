equation:

St​=S*{t-1}​+rS*{t−1}​(1−St−1​/K)+g(at​)+γpulset​+εt​

Here's yours:
Params: {'r': 0.0, 'K': 190.99527822543163, 'beta': 0.0, 'theta': nan, 'gamma': 0.0, 'lambda': 0.7937005259840998}

Based on your data, here's what we estimate:

r - growth rate: 0.0
K - carrying capacity: 200

at - adstocked spend- output of adstock(x, lam) where x = df[ad_col] (or zeros if no ads) and lam = fit["lam"] (or a fixed value from half-life)

g(at) — These are things that have a persistent effect. You started writing full time. Your are listed on a famous blog in a blogroll.

γpulset - these are transient shocks. your piece goes viral on Twitter, Substack itself features you on their homepage, you get a hacket job in the nytimes. You bought a one-time superbowl commercial for your substack. These things have a relatively short half-life.

OK, so the code should look at the raw data and try to find shocks and changes in growth rate. It throws down a bar at all of those.

Then, you have to go label each of these. What happened? Where they things you did differently, or things that happened to you?

We also need to look for changes in YOUR growth function. The computer could find these. These are going to be important because this is the only time we made the function piecewise.

Then, we've got the shocks labeled, and any changes in growth found. Then, the computer estimates the growth rate for the different time periods. (At the moment, we keep K fixed. In the future, we can change it).

Then, we display our whole equation.





Then we need to figure out the benefits of our adspend. To do that, we need to find beta, theta, and lambda.


