import pickle
import argparse

def profile(args):

    f_read = open(f'profile_{args.filename}.pkl', 'rb')

    profile_conv = pickle.load(f_read)
    f_read.close()

    if args.details:
        print(len(profile_conv))
        for i in profile_conv:
            print(i, profile_conv[i])
    else:
        profile_conv_group = {}
        for i in profile_conv:
            if i[1] not in profile_conv_group:
                profile_conv_group[i[1]] = profile_conv[i]
            else:
                profile_conv_group[i[1]] += profile_conv[i]
        for i in profile_conv_group:
            print(profile_conv_group[i])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='conv', help='you can choose conv, linear or attention')
    parser.add_argument('--details', type=int, default=0, help='see the profile details by setting it to 1')
    args = parser.parse_args()
    profile(args)
