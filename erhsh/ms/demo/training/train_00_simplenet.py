import erhsh.ms as ems


def main():
    ems.Trainer(name="simplenet") \
        .set_dataset(ems.create_mocker_dataset()) \
        .set_network(ems.SimpleNet()) \
        .set_callbacks([ems.TimeMonitor()]) \
        .run()


if __name__ == '__main__':
    main()
