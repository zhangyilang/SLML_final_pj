%% Clear
clear
clc
close all

%% Load data
train_data = load('data/train_dataset.txt');
test_data = load('data/test_dataset.txt');

%% Plot CDF
S = sigmoid(-6.25:0.01:6.25);
n = 0:50/1250:50;
data = [train_data;test_data];
TES = data(:, 2);
PCS = data(:, 3);
figure(1)
subplot(1, 2, 1)
hold on
plot(n, S)
cdfplot(TES)
axis([0, 50, 0, 1])
legend('TES', 'sigmoid')
axis square

subplot(1, 2, 2)
hold on
plot(n, S)
cdfplot(PCS)
axis([0, 50, 0, 1])
legend('PCS', 'sigmoid')
axis square
