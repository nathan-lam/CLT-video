from manim import *
import numpy as np
#from Histogram_helper import *
from Constants_helper import *

'''
Central Limit Theorem Video

Chapters
Intro
    - Intro
Simulation
    - Simulation
Example
    - Example
Moments
    - Moments
MGF
    - Moment Generating Function
MGF2
    - Properties of MGF
NormalMGF
    - Deriving Standard Normal MGF
Proof
    - Find MGF of sum of RV, show it converges to standard normal MGF
'''

class Intro(Scene):
    def construct(self):

        dice = VGroup(
            Tex('X'),
            MathTex(r'\text{Sum of }', r'1', r'\text{ d6}'),
            MathTex(r'7 = 1 + 6 = 2 + 5 = 3 + 4')
        ).arrange(DOWN)

        more_dice = VGroup(
            MathTex(r'X_1',r' + ', r'X_2'),
            MathTex(r'\text{Sum of }', r'2', r'\text{ d6}'),
            MathTex(r'X_1', r' + ', r'X_2', r'+ ... +', r'X_n'),
            MathTex(r'\text{Sum of }', r'n', r'\text{ d6}'),
        )

        #rolling 1 die
        self.play(
            FadeIn(dice[0], shift=DOWN)
        )
        self.wait()
        self.play(
            Write(dice[1])
        )
        self.wait()

        #wiggle 1 die
        self.play(
            Wiggle(dice[0],
                   scale_value=2, rotation_angle=TAU, n_wiggles=1),
        )
        self.wait()

        #rolling 2 dice
        more_dice[0].move_to(dice[0].get_center())
        more_dice[1].move_to(dice[1].get_center())
        self.play(

            ReplacementTransform(dice[0], more_dice[0]),
            TransformMatchingTex(dice[1], more_dice[1],
                                 key_map = {'1':'2'}
                                 ),
            Flash(more_dice[1][1])
        )
        self.wait()

        #wiggle 2 dice
        self.play(
            Wiggle(more_dice[0][0], #X_1
                   scale_value=2, rotation_angle=TAU, n_wiggles=1),
            Wiggle(more_dice[0][2], #X_2
                   scale_value=2, rotation_angle=TAU, n_wiggles=1),
        )
        self.wait()

        self.play(FadeIn(dice[2], shift=DOWN))
        self.wait()

        self.play(FadeOut(dice[2], shift=DOWN))
        self.wait()

        #rolling n dice
        more_dice[2].move_to(dice[0].get_center())
        more_dice[3].move_to(dice[1].get_center())
        self.play(
            ReplacementTransform(more_dice[0],more_dice[2]),
            TransformMatchingTex(more_dice[1], more_dice[3],
                                 key_map={'2': 'n'}
                                 ),
            Flash(more_dice[3][1])
        )
        self.wait()

        self.play(
            Wiggle(more_dice[2][0], #X_1
                   scale_value=2, rotation_angle=TAU, n_wiggles=1),
            Wiggle(more_dice[2][2], #X_2
                   scale_value=2, rotation_angle=TAU, n_wiggles=1),
            Wiggle(more_dice[2][4], #X_n
                   scale_value=2, rotation_angle=TAU, n_wiggles=1),
        )
        self.wait()
    #end




class Simulation(Scene):
    def construct(self):
        variables = [2, 10, 100, 500, 1000]
        batch_size = 10000

        # label of number of samples
        rv = ValueTracker(1)
        num_label = VGroup(
            Tex("Number of dice: ")
        )
        num_sample = always_redraw(lambda: DecimalNumber(num_decimal_places=0)
                                   .set_value(rv.get_value())
                                   .next_to(num_label))
        label = VGroup(num_label, num_sample)
        label.to_edge(UP)

        axes1, graph1, points1 = self.get_graph(1, batch_size)

        self.play(
            DrawBorderThenFill(axes1),
            Write(label)
        )
        self.play(
            Create(graph1),
            ShowIncreasingSubsets(points1, run_time=1))
        # rv += 1 #self.play(rv.animate.set_value(5)) to change continuously
        self.wait()

        for var in variables:
            axes2, graph2, points2 = self.get_graph(var, batch_size)
            self.play(FadeOut(points1), )
            self.wait()
            self.play(
                rv.animate.set_value(var),
                ReplacementTransform(axes1, axes2),
                ReplacementTransform(graph1, graph2),
                ShowIncreasingSubsets(points2, run_time=1)
            )
            axes1, graph1, points1 = axes2, graph2, points2
            self.wait()

        self.play(FadeOut(points1))


    def get_graph(self, num_rv, size):
        # inner_ is refering to inside this function
        # data parameters
        batch_size = size
        n_bins = 1 + 5 * num_rv
        b, a = 6, 1
        mu = num_rv * (b + a) / 2  # uniform mean
        sd = (num_rv * ((b - a + 1) ** 2 - 1) / 12) ** 0.5  # uniform sd

        # generated data of sums of uniform
        #rolls a bunch of uniform adds them up, then counts them in a dictionary
        frequency = self.CountFreq({}, self.get_unif_sample(batch_size, num_rv))
        #table of measured count (values()) with the given sum (keys())
        data = np.array([list(frequency.keys()), np.array(list(frequency.values()), dtype=float)])
        #theoretical curve
        gaussian = self.get_normal_func(mu, sd)

        #axis measurements
        xx = [min(data[0]), max(data[0])]  # boundaries for x-axis
        xxh = (xx[1] - xx[0]) / 10
        yy = [0, max(max(data[1]/batch_size), gaussian(mu))]
        yyh = yy[1] / 5

        # adding in main graph
        inner_coor = Axes(x_range=[xx[0], xx[1], xxh], y_range=[0, yy[1], yyh],
                          x_length=10, y_length=5,
                          axis_config={"include_tip": True},
                          x_axis_config={"decimal_number_config": {"num_decimal_places": 1},
                                         "number_scale_value": 0.6},
                          y_axis_config={"decimal_number_config": {"num_decimal_places": 4}}
                          ).add_coordinates()
        inner_coor_labels = inner_coor.get_axis_labels(x_label="X", y_label="Density")
        inner_axes = VGroup(inner_coor, inner_coor_labels)

        # adding in gaussian curve
        inner_graph = inner_coor.get_graph(gaussian,
                                           x_range=xx,
                                           color=RED)

        # adding in points
        inner_points = VGroup(*[Dot(inner_coor.c2p(i, j))
                                for i, j in zip(data[0], data[1] / batch_size)])
        # c2p = coordinates to points = manim coor to axes coor

        inner_axes.to_edge(DOWN)

        return inner_axes, inner_graph, inner_points

    def get_normal_func(self, mu, sd):
        #returns the pdf of Normal(mu, sd^2)
        gauss = lambda x: (2 * np.pi * sd * sd)**(-0.5) * np.exp(-0.5 * ((x - mu) / sd)**2)
        return gauss

    def get_unif_sample(self, m, n):
        # generates m samples of summing n uniform
        # n = sample/batch size
        # m = number of samples/batches
        return np.dot(np.random.randint(1, 7, size=(m, n)), np.ones(n, dtype=int))

    def CountFreq(self, prior, array):
        # prior caries the frequencies from previous data
        freq = prior  # dictionary of counts
        data = list(array)  # list of observations
        for ele in data:
            if ele in freq:
                freq[ele] += 1
            else:
                freq[ele] = 1
        return freq
    # end

class Example(Scene):
    def construct(self):
        pass
    # end

class Moments(Scene):
    def construct(self):
        pass
    # end

class MGF(Scene):
    def construct(self):
        pass
    # end

class MGF2(Scene):
    def construct(self):
        pass
    # end

class NormalMGF(Scene):
    def construct(self):
        pass
    # end

class Proof(Scene):
    def construct(self):
        pass
    # end