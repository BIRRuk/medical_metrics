# Medical Metrics


## usage

``` python
# copy the medical_metrics subfolder to your project folder

>>> import medical_metrics as mm

>>> metric = mm.BCEMed()
>>> batch_accuracy, batch_size = metirc(x, y)
>>> batch_accuracy
.75

>>> cumulative_accuracy = metric.stats(mode='val')
>>> cumulative_accuracy

acc:{class_:acc},
sensetivity:{class_:sens},
specificty:{class_:spec},
dice_score:{class_:dice}

>>> metric.print_stats()
('acc_val', values)
('sensetivity', values)
('specificity', values)
('dice', values)
('accs', values_per_class)
('senss', values_per_class)
('specs', values_per_class)
('dices', values_per_class)

```
