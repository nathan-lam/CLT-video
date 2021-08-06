from manim import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#python -m manim -p -ql CLT.py Claim
from manim import Text

#done
class Claim(Scene):
    def construct(self):

        start = VGroup(
            MathTex(r'\text{Central Limit Theorem}'),
            MathTex(r'\text{if } X_1,X_2,X_3...,X_n \overset{iid}{\sim} F'),
            MathTex(r"\text{with } mean = \mu \text{ and } variance = \sigma^2 < \infty"),
            MathTex(r'\text{then } \lim_{n \to \infty} \frac{\bar{X}-\mu}{\frac{\sigma}{\sqrt{n}}} \overset{d}{\rightarrow} Normal(0,1)')
        ).arrange(DOWN, buff=1).to_corner(UP)

        #IRL example
        example = VGroup(
            MathTex(r'X', r' = ', r'\text{\# rolled on a d6}'),
            MathTex(r'X_1',', X_2 , X_3 , ... , X_n'),
            MathTex(r'\bar{X}', r'=', r'\frac{X_1 + X_2 + X_3 + ... + X_n}{n}'),
            MathTex(r'\text{big n} \rightarrow ', r'\bar{X}', r'\sim\text{Normal}'),
            MathTex(r'\text{Independent and }', r'\text{Identically }', r'\text{Distributed}'),
            MathTex(r'I',r'I',r'D')
        )

        self.play(
            FadeIn(start, shift=DOWN)
        )
        self.wait()

        self.play(FadeOut(start[1:]))
        self.play(
            FadeIn(example[0])
        )
        self.wait()

        self.play(
            ReplacementTransform(example[0],example[1])
        )
        self.wait()

        self.play(
            ReplacementTransform(example[1],example[2])
        )
        self.wait()

        example[3].next_to(example[2].get_center()+DOWN + 2.5*LEFT)
        self.play(
            Write(example[3])
        )
        self.wait()

        #IID
        example[4].move_to(1.75*DOWN)
        for i in range(3):
            self.play(FadeIn(example[4][i], shift=DOWN))
        self.wait()

        example[5].move_to(1.75*DOWN)
        self.play(ReplacementTransform(example[4],example[5]))
        self.wait()

        self.play(
            FadeOut(start[0]),
            FadeOut(example[1:])
        )
        self.wait()


#not used, only here for future reference for barchart stuff
class Barchart_test(Scene):
    CONFIG = {
        "height":5, #sets height of axes (top to bottom)
        "width":12,
        "n_ticks":3, #number of ticks on y-axis
        "tick_width":0.2, #how wide the ticks
        "label_y_axis":True,
        "y_axis_label_height":0.25, #how big the label on the ticks
        #"max_value":8, #max(y) on chart
        "bar_fill_opacity":0.8,
        "bar_stroke_width": 4, #width of the bar's outline
        #"bar_names":["1","2","3","4","5","6"],
        "bar_label_scale_val": 0.5, #how big the labels on the x axis
    }
    def construct(self):
        #simulating the the progression of sum of RV converging to normal with barcharts
        #scrapped to use a scatter plot representation instead
        n_animations = 100
        n_batches = 50
        batch_size = 100

        counts = {}
        chart1 = BarChart(values=[],
                          max_value = 1,
                          bar_names = list(counts),
                          **self.CONFIG).scale(0.8)

        self.play(
            Write(chart1)
        )

        for _ in range(n_animations):
            counts = self.CountFreq(counts,
                                    self.get_unif_sample(n=batch_size,m=n_batches))
            chart2 = BarChart(values=list(counts.values()),
                              max_value=max(counts.values()),
                              bar_names = sorted(list(counts)),
                              **self.CONFIG).scale(0.8)

            chart2.move_to(chart1.get_center())
            self.play(
                ReplacementTransform(chart1, chart2),
                run_time = 0.01
            )
            chart1.remove()
            chart1 = chart2


    def get_unif_sample(self, n, m):
        # generates m samples of summing n uniform
        # n = number of samples or batch size
        # m = number of batches
        return np.dot(np.random.randint(1, 7, size=(m, n)), np.ones(n, dtype=int))

    def CountFreq(self, prior, array):
        # prior caries the frequencies from previous data
        freq = prior #dictionary of counts
        data = list(array) #list of observations
        for ele in data:
            if ele in freq:
                freq[ele] += 1
            else:
                freq[ele] = 1
            return freq

    #end

#done
class Simulation(Scene):
    def construct(self):

        variables = [2, 10, 100, 500]
        batch_size = 1000

        #label of number of samples
        rv = ValueTracker(1)
        num_label = VGroup(
            Tex("Number of samples: ")
        )
        num_sample = always_redraw(lambda: DecimalNumber(num_decimal_places=0)
                                   .set_value(rv.get_value())
                                 .next_to(num_label))
        label = VGroup(num_label,num_sample)
        label.to_edge(UP)

        axes1, graph1, points1 = self.get_graph(1)

        self.play(
            DrawBorderThenFill(axes1),
            Write(label)
        )
        self.play(
            Create(graph1),
            ShowIncreasingSubsets(points1, run_time = 2))
        #rv += 1 #self.play(rv.animate.set_value(5)) to change continuously
        self.wait()


        for var in variables:
            axes2, graph2, points2 = self.get_graph(var)
            self.play(FadeOut(points1),)
            self.wait()
            self.play(
                rv.animate.set_value(var),
                ReplacementTransform(axes1,axes2),
                ReplacementTransform(graph1,graph2),
                ShowIncreasingSubsets(points2, run_time = 2)
            )
            axes1, graph1, points1 = axes2, graph2, points2
            self.wait()

        self.play(FadeOut(points1))

        # #simulating points over time
        # num_sim = 10
        # new_count = {}
        # data = self.get_unif_sample(m=batch_size, n=variables[-1])
        #
        #
        # # for n in range(num_sim):
        # #     new_count = self.CountFreq(new_count, data[(100*n):(99+100*n)])
        # #     xy = np.array([list(new_count.keys()), np.array(list(new_count.values()),dtype=float)])
        # #     sim_points2 = VGroup(*[Dot(axes1[0].c2p(i, j))
        # #                           for i, j in zip(xy[0], xy[1]/batch_size)])
        # #     self.play(
        # #         ReplacementTransform(sim_points1,sim_points2)
        # #     )
        # #     sim_points1 = sim_points2
        # #     self.wait()


    def get_graph(self, num_rv):
        # inner_ is refering to inside this function
        # data parameters
        batch_size = 1000
        n_bins = 1+5*num_rv
        b, a = 6, 1
        mu = num_rv * (b + a) / 2 #uniform mean
        sd = (num_rv * ((b - a + 1) ** 2 - 1) / 12) ** 0.5 #uniform sd

        # generated data of sums of uniform
        #data = plt.hist(self.get_unif_sample(m=batch_size, n=num_rv),bins=n_bins,density=True)
        frequency = self.CountFreq({},self.get_unif_sample(batch_size,num_rv))
        data = np.array([list(frequency.keys()), np.array(list(frequency.values()),dtype=float)])
        gaussian = self.get_normal_func(mu, sd)

        xx = [min(data[0]), max(data[0])]  # boundaries for x-axis
        xxh = (xx[1] - xx[0]) / 10
        yy = [0, max(max(data[1]/batch_size), gaussian(mu))]
        yyh = yy[1]/5

        # adding in main graph
        inner_coor = Axes(x_range=[xx[0], xx[1], xxh], y_range=[0, yy[1], yyh],
                          x_length=10, y_length=5,
                          axis_config={"include_tip": True},
                          x_axis_config = {"decimal_number_config": {"num_decimal_places": 1},
                                         "number_scale_value":0.6},
                          y_axis_config = {"decimal_number_config": {"num_decimal_places": 4}}
                          ).add_coordinates()
        inner_coor_labels = inner_coor.get_axis_labels(x_label="X", y_label="Density")
        inner_axes = VGroup(inner_coor, inner_coor_labels)

        # adding in gaussian curve
        inner_graph = inner_coor.get_graph(gaussian,
                                x_range=xx,
                                color=RED)

        # adding in points
        inner_points = VGroup(*[Dot(inner_coor.c2p(i, j))
                                for i, j in zip(data[0], data[1]/batch_size)])
        #c2p = coordinates to points = manim coor to axes coor

        inner_axes.to_edge(DOWN)

        return inner_axes, inner_graph, inner_points



    def get_normal_func(self, mu, sd):
        gauss = lambda x: pow(2*np.pi*sd*sd,-0.5)*np.exp(-0.5*pow((x-mu)/sd, 2))
        return gauss


    def get_unif_sample(self, m, n):
        # generates m samples of summing n uniform
        # n = number of samples or batch size
        # m = number of batches
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




class MGF(Scene):
    #introduce what moments are
    def construct(self):
        title = Text("Moments")
        m1a, m1b = Text("Average"), MathTex(r'E(X)')
        m2a, m2b = Text("Variance"), MathTex(r'E[(X-\mu)^2]')
        m3a, m3b = Text("Skewness"), MathTex(r'E[(X-\mu)^3]')
        m4a, m4b = Text("Kurtosis"), MathTex(r'E[(X-\mu)^4]')
        E = MathTex(r'E(X^k)')
        moment = MathTex(r'=',r'\sum^\infty_{i=-\infty}',r'X^k f_X(X)', #0,1,2
                         r'\int^\infty_{-\infty}',r'X^k f_X(X)',r'dx') #3,4,5
        mgf = MathTex(r'\text{Moment Generating Function}')
        exp1 = MathTex(r'M_X(t)', r'=', r'1',r'+', r'E(X)t', r'+',
                       r'\frac{E(X^2)t^2}{2!}',
                       r'+', r'\frac{E(X^3)t^3}{3!}', r'+', r'\ldots')
        exp2 = MathTex(r'M_X(t)', r'=', r'\frac{1}{0!}', r'+', r'\frac{E(X)t}{1!}', r'+',
                       r'\frac{E(X^2)t^2}{2!}', r'+',
                       r'\frac{E(X^3)t^3}{3!}', r'+', r'\ldots')
        exp3 = MathTex(r'M_X(t)', r'=', r'E(', r'\frac{1}{0!}', r'+', r'\frac{Xt}{1!}', r'+',
                       r'\frac{(Xt)^2}{2!}', r'+',
                       r'\frac{(Xt)^3}{3!}', r'+', r'\ldots', r')')
        exp4 = MathTex(r'M_X(t)', r'=', r'E(', r'e^{Xt}', r')')
        dmgf1 = MathTex(r'\frac{d}{dt^k}', r'(',r'M_X(t)', r'=', r'E(', r'e^{Xt}', r')', r')')
        dmgf2 = MathTex(r'\frac{d}{dt^k}', r'M_X(t)', r'=', r'E(', r'X^k', r'e^{Xt}', r')')
        dmgf3 = MathTex(r'\frac{d}{dt^k}', r'M_X(0)', r'=', r'E(', r'X^k', r'e^{0}', r')')
        dmgf4 = MathTex(r'\frac{d}{dt^k}', r'M_X(0)', r'=', r'E(', r'X^k', r')')


        ma = VGroup(m1a, m2a, m3a, m4a).arrange(DOWN)
        mb = VGroup(m1b, m2b, m3b, m4b).arrange(DOWN)


        # bring in title
        self.play(Write(title))
        self.wait()

        # move title to corner
        title2 = title.copy().scale_in_place(0.5)
        title2.to_corner(UP + LEFT)
        self.play(Transform(title, title2))
        self.wait(0.5)

        #add in example moments
        self.play(FadeIn(ma, shift=DOWN))
        self.wait()

        self.play(
            ReplacementTransform(ma,mb)
        )
        self.wait()

        #start defining moments
        self.play(
            ReplacementTransform(mb, E[0])
        )
        self.wait()

        E2 = E.copy().move_to(2*LEFT)
        moment.next_to(E2.get_center() + 0.65*RIGHT) #next to LHS
        self.play(
            ReplacementTransform(E[0],E2),
            FadeIn(moment[0:3], shift=DOWN)
        )
        self.wait()

        #also showing integral definition
        moment[3:].next_to(moment[0].get_center() + 0.155*RIGHT) #next to '='
        self.play(
            ReplacementTransform(moment[1], moment[3]), #sum to integral
            ReplacementTransform(moment[2], moment[4]), #integrand stuff
            FadeIn(moment[5]) #bring in dx
        )
        self.wait()

        #introducing MGF
        self.play(
            FadeOut(E2, shift=UP),
            FadeOut(moment, shift=UP)
        )
        self.wait()

        #mgf.to_corner(UP + LEFT)
        # introducing exp
        mgf.scale_in_place(0.7)
        mgf.move_to(title2.get_center() + 1.5*RIGHT)
        self.play(
            Write(exp1),
            ReplacementTransform(title, mgf))
        self.wait()

        self.play(ReplacementTransform(exp1,exp2))
        self.wait()

        self.play(
            ReplacementTransform(exp2[   0], exp3[   0]), #M_X(t) -> M_X(t)
            ReplacementTransform(exp2[   1], exp3[ 1:3]), # '=' -> '= E('
            ReplacementTransform(exp2[2:10], exp3[3:11]),
            ReplacementTransform(exp2[  10], exp3[11: ])  # '...' -> '...)'
        )
        self.wait()

        self.play(
            ReplacementTransform(exp3[  :2], exp4[:2]), #'M_X(t) =' -> 'M_X(t) ='
            ReplacementTransform(exp3[   2], exp4[ 2]), # 'E(' -> 'E('
            ReplacementTransform(exp3[3:12], exp4[ 3]), #series to e^{Xt}
            ReplacementTransform(exp3[  -1], exp4[-1])  #')' -> ')'
        )
        self.wait()

        #adding in the derivative
        self.play(
            FadeIn(dmgf1[:2], shift=RIGHT),
            FadeIn(dmgf1[-1], shift=LEFT),
            ReplacementTransform(exp4,dmgf1[2:7])
        )
        self.wait()

        self.play(
            ReplacementTransform(dmgf1[  0], dmgf2[  0]), #'d/dt' -> 'd/dt'
            ReplacementTransform(dmgf1[  1], dmgf2[  4]), # '(' -> 'X'
            ReplacementTransform(dmgf1[2:5], dmgf2[1:4]), #'M_X(t) = E(' -> 'M_X(t) -> E('
            ReplacementTransform(dmgf1[5:7], dmgf2[5: ]), #'e^{Xt}' -> 'e^{Xt}'
            ReplacementTransform(dmgf1[ -1], dmgf2[  4])  # ')' -> 'X'
        )
        self.wait()

        #plugging in t=0
        self.play(ReplacementTransform(dmgf2, dmgf3))
        self.wait()


        self.play(
            FadeOut(dmgf3[5], shift=UP), #remove e^0
            dmgf3[-1].animate.shift(0.45*LEFT)
        )
        self.wait()
        #end

class MGF2(Scene):
    #part 2 of properties of MGF
    def construct(self):
        title = MathTex(r'\text{Moment Generating Function}').scale_in_place(0.7)

        product = VGroup(
            MathTex(r'M_A(t)M_B(t) = ', r'E(', r'e^{At})E(e^{Bt}', r')'),
            MathTex(r'M_A(t)M_B(t) = ', r'E(', r'e^{At}e^{Bt}', r')'),
            MathTex(r'M_A(t)M_B(t) = ', r'E(', r'e^{At+Bt}', r')'),
            MathTex(r'M_A(t)M_B(t) = ', r'E(', r'e^{(A+B)t}', r')'),
            MathTex(r'M_A(t)M_B(t) = ', r'M_{A+B}(t)')
        )
        converge = VGroup(
            MathTex(r'M_{A_n}(t)',r'\rightarrow',r'M_B(t)'),
            MathTex(r'A_n',r'\overset{d}{\rightarrow}',r'B')
        ).arrange(DOWN)


        title.to_corner(UP + LEFT)
        self.add(title)

        #introduce product property
        self.play(
            FadeIn(product[0], shift=DOWN)
        )
        self.wait()

        self.remove(product[0])
        #factor out expectation
        #product2.move_to(product1[1].get_center() + 0.4*LEFT)
        self.play(
            TransformMatchingTex(product[0],
                                 product[1],
                                 transform_mismatches=True,
                                 key_map= dict({"e^{At})E(e^{Bt}": "e^{At}e^{Bt}"})
                                 )
        )

        self.wait()


        #combine exponents
        self.play(
            TransformMatchingTex(product[1],
                                 product[2],
                                 transform_mismatches=True,
                                 key_map=dict({"e^{At}e^{Bt}":"e^{At+Bt}"})
                                 )
        )
        self.wait()


        # #factor exponent
        self.play(
            TransformMatchingTex(product[2],
                                 product[3],
                                 transform_mismatches=True,
                                 key_map=dict({"e^{At+Bt}":"e^{(A+B)t}"})
                                 )
        )
        self.wait()


        # #conclusion
        self.play(
            TransformMatchingTex(product[3],
                                 product[4],
                                 transform_mismatches=True,
                                 key_map=dict({"e^{(A+B)t}":"M_{A+B}(t)",
                                               "(":"M_{A+B}(t)",
                                               ")":"M_{A+B}(t)"})
                                 )
        )
        self.wait()

        # #clearing screen
        self.play(
            FadeOut(product[4], shift=UP),
        )
        self.wait()

        # #adding in convergence property
        self.play(FadeIn(converge[0], shift=DOWN))
        self.wait()

        self.play(TransformMatchingTex(converge[0].copy(), converge[1],
                                       key_map={
                                           "M_{A_n}(t)": "A_n",
                                           r"\rightarrow": r"\overset{d}{\rightarrow}",
                                           "M_B(t)": "B"}
                                       )
                  )
        self.wait()


#done animating algebra
class NormalMGF(Scene):
    #derive the standard normal MGF
    def construct(self):
        start1 = Text("Standard Normal MGF")
        LHS = MathTex(r'M_Z(t) = ')
        RHS1 = MathTex(r'E(e^{xt})')
        RHS2 = MathTex(r'\int^\infty_{-\infty}', r'e^{xt}',r'f_X(x)dx')
        RHS3 = MathTex(r'\frac{1}{\sqrt{2\pi}}', r'e^{-\frac{x^2}{2}}dx')
        RHS4 = MathTex(r'\frac{1}{\sqrt{2\pi}}', r'e^{xt-\frac{x^2}{2}}dx')
        RHS5 = MathTex(r'e^{(\frac{t^2}{2}', r'-\frac{t^2}{2}) + xt-\frac{x^2}{2}}dx')
        RHS6 = MathTex(r'e^{\frac{t^2}{2}', r'-(\frac{t^2}{2} + xt-\frac{x^2}{2})}dx')
        RHS7 = MathTex(r'e^{\frac{t^2}{2}', r'-\frac{1}{2}(x-t)^2}dx')
        RHS8 = MathTex(r'e^{\frac{t^2}{2}}', r'e^{-\frac{1}{2}(x-t)^2}dx')
        RHS9 = MathTex(r'e^{\frac{t^2}{2}}', r'\int^\infty_{-\infty}',
                       r'\frac{1}{\sqrt{2\pi}}', r'e^{-\frac{1}{2}(x-t)^2}dx')
        line10 = MathTex(r'M_Z(t) = ', r'e^{\frac{t^2}{2}}')


        #bring in title
        self.play(Write(start1))
        self.wait()

        #move title to corner
        start2 = start1.copy().scale_in_place(0.5)
        start2.to_corner(UP + LEFT)
        self.play(Transform(start1, start2))
        self.wait(0.5)

        #starting algebra, line 1
        LHS.move_to(2*LEFT)
        RHS1.next_to(LHS.get_center()+RIGHT)
        self.play(
            FadeIn(LHS, shift=DOWN),
            FadeIn(RHS1, shift=DOWN)
        )
        self.wait()

        #start line 2
        RHS2.next_to(LHS.get_center() + RIGHT)
        self.play(ReplacementTransform(RHS1,RHS2))
        self.wait()

        #start line 3
        RHS3.next_to(RHS2[1].get_center() + 0.2*RIGHT) #next to \int^\infty_{-\infty}'
        self.play(ReplacementTransform(RHS2[2], RHS3))
        self.wait()

        #start line 4
        RHS4.next_to(RHS2[0].get_center() + 0.2 * RIGHT) #next to \frac{1}{\sqrt{2\pi}}
        self.play(
            ReplacementTransform(RHS2[1], RHS4[1]),
            ReplacementTransform(RHS3[1], RHS4[1]),
            ReplacementTransform(RHS3[0], RHS4[0])
        )
        self.wait()

        #start line 5
        RHS5.next_to(RHS4[0].get_center() + 0.4*RIGHT) #next to \frac{1}{\sqrt{2\pi}}
        self.play(ReplacementTransform(RHS4[1], RHS5))
        self.wait()

        # start line 6
        RHS6.next_to(RHS4[0].get_center() + 0.4 * RIGHT)#next to \frac{1}{\sqrt{2\pi}}
        self.play(
            ReplacementTransform(RHS5[0], RHS6[0]),
            ReplacementTransform(RHS5[1], RHS6[1])
        )
        self.wait()

        # start line 7
        RHS7.next_to(RHS4[0].get_center() + 0.4 * RIGHT)  # next to \frac{1}{\sqrt{2\pi}}
        self.play(
            ReplacementTransform(RHS6[0], RHS7[0]),
            ReplacementTransform(RHS6[1], RHS7[1])
        )
        self.wait()

        # start line 8
        RHS8.next_to(RHS4[0].get_center() + 0.4 * RIGHT)  # next to \frac{1}{\sqrt{2\pi}}
        self.play(
            ReplacementTransform(RHS7[0], RHS8[0]),
            ReplacementTransform(RHS7[1], RHS8[1])
        )
        self.wait()

        # start line 9
        RHS9.next_to(LHS.get_center() + RIGHT)  # next to \frac{1}{\sqrt{2\pi}}
        self.play(
            ReplacementTransform(RHS8[0], RHS9[0]),
            ReplacementTransform(RHS2[0], RHS9[1]),
            ReplacementTransform(RHS4[0], RHS9[2]),
            ReplacementTransform(RHS8[1], RHS9[3])
        )
        self.wait()

        # start line 10
        self.play(FadeOut(RHS9[1:], shift=UP))
        self.wait(0.2)

        self.play(
            ReplacementTransform(LHS, line10[0]),
            ReplacementTransform(RHS9[0], line10[1])
        )
        self.wait()

        #end




class Sums(Scene):

    def construct(self):
        title = MathTex(r'\text{Proof}')

        S = VGroup(
            MathTex(r'S'),
            MathTex(r'S', r'=', r'X_1 + X_2 + X_3 + ... + X_n'),
            MathTex(r'S^*', r'=',r'\frac{S-E(S)}{\sqrt{Var(S)}}'),
            MathTex(r'S^*', r'=',r'\frac{S - n\mu}{\sqrt{n\sigma^2}}'),
            MathTex(r'S^*', r'=',r'\frac{\sum^n_{i=1} X_i - \sum^n_{i=1} \mu}{\sqrt{n\sigma^2}}'),
            MathTex(r'S^*', r'=',r'\frac{\sum^n_{i=1} (X_i - \mu)}{\sqrt{n\sigma^2}}'),
            MathTex(r'M_{S^*}(t)', r'=',r'E(',r'e^{S^*t}', r')'),
            MathTex(r'M_{S^*}(t)', r'=', r'E(', r'e^{\frac{\sum^n_{i=1} (X_i - \mu)}{\sqrt{n\sigma^2}}t}', r')'),
            MathTex(r'M_{S^*}(t)', r'=', r'E(', r'\prod\limits^{n}_{i=1}e^{\frac{(X_i - \mu)}{\sqrt{n\sigma^2}}t}', r')'),
            MathTex(r'M_{S^*}(t)', r'=',  r'\prod\limits^{n}_{i=1}', r'E(',r'e^{\frac{(X_i - \mu)}{\sqrt{n\sigma^2}}t}', r')'),
            MathTex(r'M_{S^*}(t)', r'=', r'E(', r'(e^{(X - \mu)\frac{t}{\sqrt{n\sigma^2}}})^n', r')'),
            MathTex(r'M_{S^*}(t)', r'=', r'[M_{X-\mu}(\frac{t}{\sqrt{n\sigma^2}})]^n'),
            MathTex(r'M_{S^*}(t)', r'=',
                    r'[\frac{M_{X-\mu}(\frac{t}{\sqrt{n\sigma^2}} = a)}{0!} + '
                    r'\frac{d}{dt}\frac{M_{X-\mu}(\frac{t}{\sqrt{n\sigma^2}} = a)}{1!}(\frac{t}{\sqrt{n\sigma^2}}-a) \\ + '
                    r'\frac{d}{dt^2}\frac{M_{X-\mu}(\frac{t}{\sqrt{n\sigma^2}} = a)}{2!}(\frac{t}{\sqrt{n\sigma^2}}-a)^2 + '
                    r'\frac{d}{dt^3}\frac{M_{X-\mu}(\frac{t}{\sqrt{n\sigma^2}} = a)}{3!}(\frac{t}{\sqrt{n\sigma^2}}-a)^3 + ...]^n'
                    ).scale_to_fit_width(11),
            MathTex(r'M_{S^*}(t)', r'=',
                    r'[\frac{M_{X-\mu}(0)}{0!} + '
                    r'\frac{d}{dt}  \frac{M_{X-\mu}(0)}{1!}(\frac{t}{\sqrt{n\sigma^2}}) \\ +'
                    r'\frac{d}{dt^2}\frac{M_{X-\mu}(0)}{2!}(\frac{t}{\sqrt{n\sigma^2}})^2 + '
                    r'\frac{d}{dt^3}\frac{M_{X-\mu}(0)}{3!}(\frac{t}{\sqrt{n\sigma^2}})^3 + ',
                    r'...]^n'
                    ).scale_to_fit_width(11),
            MathTex(r'M_{S^*}(t)', r'=',
                    r'[\frac{E(e^{(X-\mu)(0)})}{0!} + '
                    r'\frac{E((X-\mu)  e^{(X-\mu)(0)})}{1!}(\frac{t}{\sqrt{n\sigma^2}}) \\ +'
                    r'\frac{E((X-\mu)^2e^{(X-\mu)(0)})}{2!}(\frac{t}{\sqrt{n\sigma^2}})^2 + '
                    r'\frac{E((X-\mu)^3e^{(X-\mu)(0)})}{3!}(\frac{t}{\sqrt{n\sigma^2}})^3 + ',
                    r'...]^n'
                    ).scale_to_fit_width(11),
            MathTex(r'M_{S^*}(t)', r'=',
                    r'[1 + '
                    r'E(X-\mu)(\frac{t}{\sqrt{n\sigma^2}}) +'
                    r'E((X-\mu)^2)(\frac{t^2}{2n\sigma^2}) \\ + '
                    r'E((X-\mu)^3)(\frac{t^3}{6(n\sigma^2)^{3/2}}) + ',
                    r'...]^n'
                    ).scale_to_fit_width(11),
            MathTex(r'M_{S^*}(t)', r'\approx',
                    r'[1 + '
                    r'0 + '
                    r'\sigma^2(\frac{t^2}{2n\sigma^2})]^n'
                    ),
            MathTex(r'M_{S^*}(t)', r'\approx',
                    r'[1 + \frac{t^2}{2n}]^n'
                    ),
            MathTex(r'\lim_{n \to \infty} M_{S^*}(t)', r'\approx',
                    r'\lim_{n \to \infty}(1 + \frac{(\frac{t^2}{2})}{n})^n'
                    ),
            MathTex(r'\lim_{n \to \infty} M_{S^*}(t)', r'\approx',
                    r'e^{\frac{1}{2}t^2}'
                    ),
            MathTex(r'\lim_{n \to \infty} M_{S^*}(t)', r'\approx',
                    r'M_Z(t)'
                    ),



        )

        ES = VGroup(
            MathTex(r'E(S)', r'=', r'E(X_1 + X_2 + X_3 + ... + X_n)'),
            MathTex(r'E(S)', r'=', r'E(X_1) + E(X_2) + E(X_3) + ... + E(X_n)'),
            MathTex(r'E(S)', r'=', r'nE(X)'),
            MathTex(r'E(S)', r'=', r'n\mu'),
        )

        varS = VGroup(
            MathTex(r'Var(S)', r'=', r'Var(X_1 + X_2 + X_3 + ... + X_n)'),
            MathTex(r'Var(S)', r'=', r'Var(X_1) + Var(X_2) + Var(X_3) + ... + Var(X_n)'),
            MathTex(r'Var(S)', r'=', r'nVar(X)'),
            MathTex(r'Var(S)', r'=', r'n\sigma^2')
        )

        self.play(FadeIn(title))
        self.wait()

        self.play(
            ReplacementTransform(title,
                                 title.copy().to_edge(UP + LEFT))
        )
        self.wait()

        #defining S
        self.play(Write(S[0]))
        self.wait()

        self.play(
            S[0].animate.move_to(S[1][0]),
            FadeIn(S[1][1:], shift=LEFT)
        )
        self.wait()
        S2 = S[1].copy()

        #deriving E(S)
        self.play(
            ReplacementTransform(S[0], ES[0][0]), #S to E(S)
            ReplacementTransform(S[1],ES[0]) # sum rv to sum E(rv)
        )
        self.wait()

        for i in range(3):
            self.play(
                ReplacementTransform(ES[i],ES[i+1])
            )
            self.wait()

        self.play(
            ES[3].animate.to_edge(UP + RIGHT),
            FadeIn(S2)
        )


        #deriving Var(S)
        self.play(
            ReplacementTransform(S2,varS[0])
        )
        self.wait()

        for i in range(3):
            self.play(
                ReplacementTransform(varS[i], varS[i + 1])
            )
            self.wait()

        #standardized S
        self.play(
            varS[3].animate.move_to(ES[3].get_center() + DOWN),
            FadeIn(S[2])
        )
        self.wait()

        for i in range(18):
            self.play(
                ReplacementTransform(S[2+i],S[3+i])
            )
            self.wait()

