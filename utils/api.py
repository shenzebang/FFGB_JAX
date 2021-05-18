from collections import namedtuple

FedAlgorithm = namedtuple("FedAlgorithm", "sampler\
                                           server_init\
                                           client_init\
                                           client_step\
                                           vclient_step\
                                           client_end\
                                           server_step")

Classifier = namedtuple("Classifier", "params, weights, is_empty")

Batch = namedtuple("Batch", "x, y")

OracleState = namedtuple("OracleState", "lr num_steps batch_size")

ServerHyperParams = namedtuple("ExperimentOptions", "num_rounds distill_ratio lr_0 num_sampled_clients\
                                                    num_distill_rounds num_local_steps num_clients s num_classes\
                                                    oracle_num_steps, oracle_lr, oracle_batch_size\
                                                    num_channels, get_classifier_fn\
                                                    distill_oracle_num_steps, distill_oracle_lr, distill_oracle_batch_size,\
                                                    image_size")


FFGBDistillClientState = namedtuple("FFGBDistillClientState",
                                    "f_x, residual, local_step, num_local_steps, global_round, lr_0")

FFGBDistillServerState = namedtuple("FFGBDistillServerState", "classifier, global_round")

StaticFns = namedtuple("StaticFns", ['get_classifier_fn', 'model_apply_fn'])