def plot_function(bot, sim, b_plot_spikes=False):
    if bot.b_probe:
        import matplotlib.pyplot as plt
        from nengo.utils.matplotlib import rasterplot
        from matplotlib.font_manager import FontProperties

        fontP = FontProperties()
        fontP.set_size('small')

        if bot.b_base:
            plt.figure("base")
            plt.plot(sim.trange(), sim.data[bot.p_base_neurons_out])
            plt.legend(["x", "y", "z"], prop=fontP)

            if b_plot_spikes:
                # Plot the spiking output of the ensemble
                plt.figure("base spikes")
                rasterplot(sim.trange(), sim.data[bot.p_base_neurons_spikes])
        if bot.b_arm:
            plt.figure("arm")
            plt.plot(sim.trange(), sim.data[bot.p_arm_neurons_out])
            plt.legend(["shoulder", "elbow", "hand", "gripper"], prop=fontP)

            if b_plot_spikes:
                # Plot the spiking output of the ensemble
                plt.figure("arm spikes")
                rasterplot(sim.trange(), sim.data[bot.p_arm_neurons_spikes])
        if bot.b_freqs:
            for name in bot.bot.adress_list:
                if "retina" in name:
                    n_freqs = int(bot.freqs[name].get_output_dim()/3.0)
                    # FIXME: label needs fixing. the data for three tracked stimuli looks like [x1,x2,x3,y1,y2,y3,c1,c2,c3]
                    labels = ["x", "y", "certainty"]*n_freqs
                    plt.figure("freqs_" + name)
                    plt.subplot(211)
                    plt.plot(sim.trange(), sim.data[bot.p_freqs_out[name]])
                    plt.legend(labels, prop=fontP)
                    plt.subplot(212)
                    plt.plot(sim.trange(), sim.data[bot.p_freqs_neurons_out[name]])
                    plt.legend(labels, prop=fontP)

                    if b_plot_spikes:
                        # Plot the spiking output of the ensemble
                        plt.figure("freqs spikes")
                        rasterplot(sim.trange(), sim.data[bot.p_freqs_neurons_spikes])
        elif bot.b_tracker:
            if "retina" in name:
                n_freqs = int(bot.trackers[name].get_output_dim()/4.0)
                # FIXME: label needs fixing. the data for three tracked stimuli looks like [x1,x2,x3,y1,y2,y3,r1,r2,r3,c1,c2,c3]
                labels = ["x", "y", "radius", "certainty"]*n_freqs
                plt.figure("trackers_" + name)
                plt.subplot(211)
                plt.plot(sim.trange(), sim.data[bot.p_trackers_out[name]])
                plt.legend(labels, prop=fontP)
                plt.subplot(212)
                plt.plot(sim.trange(), sim.data[bot.p_trackers_neurons_out[name]])
                plt.legend(labels, prop=fontP)

                if b_plot_spikes:
                    # Plot the spiking output of the ensemble
                    plt.figure("freqs spikes")
                    rasterplot(sim.trange(), sim.data[bot.p_trackers_neurons_spikes])

        if bool(bot.b_sensors):
            sensor_labels = ["v1", "v2", "v3", "v4", "v5"]
            for k, b_sensor in bot.b_sensors.iteritems():
                if b_sensor:
                    length = getattr(bot, k).get_output_dim()
                    labels = sensor_labels[:length]
                    plt.figure(k)
                    plt.subplot(211)
                    plt.plot(sim.trange(), sim.data[getattr(bot, "p_"+k+"_out")])
                    plt.legend(labels, prop=fontP)
                    plt.subplot(212)
                    plt.plot(sim.trange(), sim.data[getattr(bot, "p_"+k+"_neurons_out")])
                    plt.legend(labels, prop=fontP)

                    if b_plot_spikes:
                        # Plot the spiking output of the ensemble
                        plt.figure(k+" spikes")
                        rasterplot(sim.trange(), sim.data[getattr(bot, "p_"+k+"_neurons_spikes")])

        plt.show()
