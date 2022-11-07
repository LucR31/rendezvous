# Spaceship rendezvous<br> with Reinforcement Learning, part 1
<hr>
<b>Description:</b> The agent, a spaceship orbiting the Earth, wants to make a rendezvous with a target spaceship: with in a fixed orbit aroud the earth. The agent has limited fuel. It will try to use it up to get its orbit as close as possible with the target orbit. <br><br>
<b>Method:</b> We use RL methods on the agent, form different initial orbits with the end of finding the target orbit as fuel efficient as possible. <br><br>

## The Environment
<hr>
Made with pygame, and using Ploiastro library.<br>

Steps of 1 minute<br>
Action in each step is an impulse $dv=[x,y,z]$ where $x,y,z$ in $[-10,10]$ that changes the velocity vector, and therefore the orbit. The agent will not stop using fuel to modify its orbit unless it matches that one of the target.<br> 
We consider three diferent inital orbits for the agent.<br>
The game ends when:
<ul>
    <li>The agent runs out of fuel</li>
    <li>The agent crashed into the earth</li>
    <li>The agent orbit is <i>perfectly</i>(with certain error margin) alinged to the targets one.</li>
</ul>

The orbits difference is measured as follows:<br>

The reward system gives one point when the orbit gets closer to the target's orbit and minus one when gets further from it.<br>

Preview:<br>


<img style="margin-left: 100px;" src="data/orb.gif" width="500" height="500"/>


## PPO
<hr>


